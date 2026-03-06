use super::Tensor;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl Tensor {
    /// Elementwise addition.
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch in add: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let mut result_data = vec![0.0; self.data.len()];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    Self::add_avx2(&self.data, &other.data, &mut result_data);
                }
            } else {
                Self::add_fallback(&self.data, &other.data, &mut result_data);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::add_fallback(&self.data, &other.data, &mut result_data);
        }

        Tensor::new(self.shape.clone(), result_data)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let mut i = 0;
        let n = a.len();
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), vc);
            i += 8;
        }
        for j in i..n {
            out[j] = a[j] + b[j];
        }
    }

    fn add_fallback(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    /// Matrix multiplication. 2D arrays only for simplicity.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matmul requires 2D tensors".to_string());
        }
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k1 != k2 {
            return Err(format!("Inner dimension mismatch: {} vs {}", k1, k2));
        }

        let mut out = vec![0.0; m * n];

        #[cfg(feature = "cuda")]
        {
            static CUDA_CELL: OnceLock<Option<CudaDevice>> = OnceLock::new();
            if let Some(device) = CUDA_CELL.get_or_init(CudaDevice::new) {
                // Try to use persistent GPU pointers if they exist
                if let (Some(ptr_a), Some(ptr_b)) = (self.device_ptr, other.device_ptr) {
                    let mut out_tensor = Tensor::zeros(vec![m, n]);
                    let ptr_c = device.alloc(m * n * 4).unwrap();
                    if device
                        .matmul_persistent(
                            m,
                            k1,
                            n,
                            ptr_a.0 as *const _,
                            ptr_b.0 as *const _,
                            ptr_c as *mut _,
                        )
                        .is_ok()
                    {
                        device.copy_d2h(&mut out_tensor.data, ptr_c).unwrap();
                        device.free(ptr_c);
                        return Ok(out_tensor);
                    }
                }

                // Fallback to one-shot matmul (which uses allocation inside)
                if device
                    .matmul(m, k1, n, &self.data, &other.data, &mut out)
                    .is_ok()
                {
                    return Ok(Tensor::new(vec![m, n], out).unwrap());
                }
            }
        }

        // Parallelize over M dimension to utilize all CPU cores
        let num_threads = rayon::current_num_threads().max(1);
        let m_chunk = std::cmp::max(1, (m + num_threads - 1) / num_threads);

        out.par_chunks_mut(m_chunk * n)
            .enumerate()
            .for_each(|(i, out_chunk)| {
                let current_m = out_chunk.len() / n;
                let a_ptr = unsafe { self.data.as_ptr().add(i * m_chunk * k1) };
                let b_ptr = other.data.as_ptr();
                let c_ptr = out_chunk.as_mut_ptr();

                unsafe {
                    matrixmultiply::sgemm(
                        current_m,
                        k1,
                        n,
                        1.0,
                        a_ptr,
                        k1 as isize,
                        1,
                        b_ptr,
                        n as isize,
                        1,
                        0.0,
                        c_ptr,
                        n as isize,
                        1,
                    );
                }
            });

        Tensor::new(vec![m, n], out)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let mut out = self.data.clone();
        for x in out.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        Tensor::new(self.shape.clone(), out).unwrap()
    }

    /// Softmax along the last dimension.
    pub fn softmax(&self) -> Result<Tensor, String> {
        if self.shape.is_empty() {
            return Err("Cannot perform softmax on scalar".to_string());
        }
        let last_dim = *self.shape.last().unwrap();
        let outer_dims: usize = self.shape[..self.shape.len() - 1].iter().product();

        let mut out = self.data.clone();
        for i in 0..outer_dims {
            let offset = i * last_dim;
            let slice = &mut out[offset..offset + last_dim];

            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in slice.iter_mut() {
                *x = (*x - max_val).exp();
                sum += *x;
            }
            if sum > 0.0 {
                for x in slice.iter_mut() {
                    *x /= sum;
                }
            }
        }

        Tensor::new(self.shape.clone(), out)
    }

    /// Layer Normalization along the last dimension.
    pub fn layernorm(&self, eps: f32) -> Result<Tensor, String> {
        if self.shape.is_empty() {
            return Err("Cannot perform layernorm on scalar".to_string());
        }
        let last_dim = *self.shape.last().unwrap();
        let outer_dims: usize = self.shape[..self.shape.len() - 1].iter().product();

        let mut out = self.data.clone();
        for i in 0..outer_dims {
            let offset = i * last_dim;
            let slice = &mut out[offset..offset + last_dim];

            let mut sum = 0.0;
            for &x in slice.iter() {
                sum += x;
            }
            let mean = sum / last_dim as f32;

            let mut var_sum = 0.0;
            for &x in slice.iter() {
                let diff = x - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / last_dim as f32;
            let std_dev = (variance + eps).sqrt();

            for x in slice.iter_mut() {
                *x = (*x - mean) / std_dev;
            }
        }

        Tensor::new(self.shape.clone(), out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_fallback() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(vec![2, 2], vec![2.0, 0.0, 1.0, 2.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.data, vec![4.0, 4.0, 10.0, 8.0]);
    }
}
