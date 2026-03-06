use std::ptr;

#[link(name = "cuda")]
#[link(name = "cudart")]
#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(handle: *mut *mut std::ffi::c_void) -> i32;
    fn cublasDestroy_v2(handle: *mut std::ffi::c_void) -> i32;
    fn cublasSgemm_v2(
        handle: *mut std::ffi::c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: *const f32,
        C: *mut f32,
        ldc: i32,
    ) -> i32;

    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaDeviceSynchronize() -> i32;

    pub fn launch_ssm_scan(
        a_seq: *const f32,
        b_seq: *const f32,
        out: *mut f32,
        h_states: *mut f32,
        batch_size: i32,
        seq_len: i32,
        dim: i32,
    );
}

pub struct CudaDevice {
    handle: *mut std::ffi::c_void,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new() -> Option<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            if cublasCreate_v2(&mut handle) == 0 {
                Some(CudaDevice { handle })
            } else {
                None
            }
        }
    }

    pub fn matmul(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), String> {
        unsafe {
            let mut d_a = ptr::null_mut();
            let mut d_b = ptr::null_mut();
            let mut d_c = ptr::null_mut();

            if cudaMalloc(&mut d_a, m * k * 4) != 0 {
                return Err("cudaMalloc A failed".into());
            }
            if cudaMalloc(&mut d_b, k * n * 4) != 0 {
                cudaFree(d_a);
                return Err("cudaMalloc B failed".into());
            }
            if cudaMalloc(&mut d_c, m * n * 4) != 0 {
                cudaFree(d_a);
                cudaFree(d_b);
                return Err("cudaMalloc C failed".into());
            }

            // H2D
            cudaMemcpy(d_a, a.as_ptr() as *const _, m * k * 4, 1);
            cudaMemcpy(d_b, b.as_ptr() as *const _, k * n * 4, 1);

            let alpha = 1.0f32;
            let beta = 0.0f32;

            cublasSgemm_v2(
                self.handle,
                0,
                0,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                d_b as *const f32,
                n as i32,
                d_a as *const f32,
                k as i32,
                &beta,
                d_c as *mut f32,
                n as i32,
            );

            // D2H
            cudaMemcpy(c.as_mut_ptr() as *mut _, d_c, m * n * 4, 2);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

            if cudaDeviceSynchronize() != 0 {
                return Err("cudaDeviceSynchronize failed".into());
            }
        }
        Ok(())
    }

    pub fn alloc(&self, size_bytes: usize) -> Result<*mut std::ffi::c_void, String> {
        let mut ptr = ptr::null_mut();
        unsafe {
            if cudaMalloc(&mut ptr, size_bytes) == 0 {
                Ok(ptr)
            } else {
                Err("cudaMalloc failed".into())
            }
        }
    }

    pub fn free(&self, ptr: *mut std::ffi::c_void) {
        unsafe {
            cudaFree(ptr);
        }
    }

    pub fn copy_h2d(&self, dst: *mut std::ffi::c_void, src: &[f32]) -> Result<(), String> {
        unsafe {
            if cudaMemcpy(dst, src.as_ptr() as *const _, src.len() * 4, 1) == 0 {
                Ok(())
            } else {
                Err("cudaMemcpy H2D failed".into())
            }
        }
    }

    pub fn copy_d2h(&self, dst: &mut [f32], src: *mut std::ffi::c_void) -> Result<(), String> {
        unsafe {
            if cudaMemcpy(dst.as_mut_ptr() as *mut _, src, dst.len() * 4, 2) == 0 {
                Ok(())
            } else {
                Err("cudaMemcpy D2H failed".into())
            }
        }
    }

    pub fn matmul_persistent(
        &self,
        m: usize,
        k: usize,
        n: usize,
        d_a: *const std::ffi::c_void,
        d_b: *const std::ffi::c_void,
        d_c: *mut std::ffi::c_void,
    ) -> Result<(), String> {
        unsafe {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            if cublasSgemm_v2(
                self.handle,
                0,
                0, // CUBLAS_OP_N
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                d_b as *const f32,
                n as i32,
                d_a as *const f32,
                k as i32,
                &beta,
                d_c as *mut f32,
                n as i32,
            ) == 0
            {
                Ok(())
            } else {
                Err("cublasSgemm failed".into())
            }
        }
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle);
        }
    }
}
