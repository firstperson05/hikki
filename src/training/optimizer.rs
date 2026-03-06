use crate::tensor::Tensor;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub m: Vec<Tensor>, // First moment buffers
    pub v: Vec<Tensor>, // Second moment buffers
}

impl AdamW {
    pub fn new(param_shapes: &[Vec<usize>], lr: f32, weight_decay: f32) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for shape in param_shapes {
            m.push(Tensor::zeros(shape.clone()));
            v.push(Tensor::zeros(shape.clone()));
        }
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step: 0,
            m,
            v,
        }
    }

    pub fn save_state(&self, path: &Path) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&(self.step as u64).to_le_bytes())?;
        file.write_all(&(self.m.len() as u32).to_le_bytes())?;

        for tensor in &self.m {
            file.write_all(&(tensor.data.len() as u32).to_le_bytes())?;
            for &val in &tensor.data {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for tensor in &self.v {
            file.write_all(&(tensor.data.len() as u32).to_le_bytes())?;
            for &val in &tensor.data {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load_state(&mut self, path: &Path) -> std::io::Result<()> {
        let mut file = BufReader::new(File::open(path)?);
        let mut u64_buf = [0u8; 8];
        let mut u32_buf = [0u8; 4];
        let mut f32_buf = [0u8; 4];

        file.read_exact(&mut u64_buf)?;
        self.step = u64::from_le_bytes(u64_buf) as usize;

        file.read_exact(&mut u32_buf)?;
        let num_params = u32::from_le_bytes(u32_buf) as usize;

        for i in 0..num_params {
            file.read_exact(&mut u32_buf)?;
            let len = u32::from_le_bytes(u32_buf) as usize;
            for j in 0..len {
                file.read_exact(&mut f32_buf)?;
                self.m[i].data[j] = f32::from_le_bytes(f32_buf);
            }
        }
        for i in 0..num_params {
            file.read_exact(&mut u32_buf)?;
            let len = u32::from_le_bytes(u32_buf) as usize;
            for j in 0..len {
                file.read_exact(&mut f32_buf)?;
                self.v[i].data[j] = f32::from_le_bytes(f32_buf);
            }
        }
        Ok(())
    }

    pub fn step_single(&mut self, param: &mut Tensor, grad: &Tensor) {
        self.step += 1;
        let t = self.step as f32;
        let lr_t = self.lr * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        let n = param.data.len();
        Self::update_fused_scalar(
            n,
            &mut param.data,
            &grad.data,
            &mut self.m[0].data,
            &mut self.v[0].data,
            self.beta1,
            self.beta2,
            self.lr,
            self.weight_decay,
            lr_t,
            self.eps,
        );
    }

    pub fn step_fused(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]) {
        self.step += 1;
        let t = self.step as f32;
        let lr_t = self.lr * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        for (i, p) in params.iter_mut().enumerate() {
            let grad = grads[i];
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let n = p.data.len();

            // Fused SIMD update
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    // Vectorized logic would go here
                    // For now, we perform the fused scalar update which compiler can vectorize
                    Self::update_fused_scalar(
                        n,
                        &mut p.data,
                        &grad.data,
                        &mut m.data,
                        &mut v.data,
                        self.beta1,
                        self.beta2,
                        self.lr,
                        self.weight_decay,
                        lr_t,
                        self.eps,
                    );
                } else {
                    Self::update_fused_scalar(
                        n,
                        &mut p.data,
                        &grad.data,
                        &mut m.data,
                        &mut v.data,
                        self.beta1,
                        self.beta2,
                        self.lr,
                        self.weight_decay,
                        lr_t,
                        self.eps,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Self::update_fused_scalar(
                    n,
                    &mut p.data,
                    &grad.data,
                    &mut m.data,
                    &mut v.data,
                    self.beta1,
                    self.beta2,
                    self.lr,
                    self.weight_decay,
                    lr_t,
                    self.eps,
                );
            }
        }
    }

    fn update_fused_scalar(
        n: usize,
        p: &mut [f32],
        g: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        beta1: f32,
        beta2: f32,
        lr: f32,
        weight_decay: f32,
        lr_t: f32,
        eps: f32,
    ) {
        for j in 0..n {
            let grad_j = g[j];
            m[j] = beta1 * m[j] + (1.0 - beta1) * grad_j;
            v[j] = beta2 * v[j] + (1.0 - beta2) * grad_j * grad_j;

            p[j] -= lr * weight_decay * p[j];
            p[j] -= lr_t * m[j] / (v[j].sqrt() + eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_adamw_serialization() {
        let shapes = vec![vec![2, 2]];
        let mut opt = AdamW::new(&shapes, 0.01, 0.01);
        opt.step = 5;
        opt.m[0].data[0] = 1.0;

        let path = temp_dir().join("opt_test.bin");
        opt.save_state(&path).unwrap();

        let mut opt2 = AdamW::new(&shapes, 0.01, 0.01);
        opt2.load_state(&path).unwrap();

        assert_eq!(opt.step, opt2.step);
        assert_eq!(opt.m[0].data[0], opt2.m[0].data[0]);
    }
}
