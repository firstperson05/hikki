#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#define __global__
#define __device__
#include <device_functions.h>
#include <device_launch_parameters.h>
#endif

#define BLOCK_SIZE 256

extern "C" __global__ void relu_kernel(float *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (data[idx] < 0.0f)
      data[idx] = 0.0f;
  }
}

extern "C" __global__ void add_kernel(float *a, const float *b, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    a[idx] += b[idx];
}

extern "C" __global__ void
ssm_scan_kernel(const float *a_seq, // (Dim) weights
                const float *b_seq, // (Batch, Seq, Dim) inputs
                float *out,         // (Batch, Seq, Dim) result
                float *h_states,    // (Batch, Dim) in/out last states
                int batch_size, int seq_len, int dim) {
  int b = blockIdx.x;
  int d = blockIdx.y * blockDim.x + threadIdx.x;

  if (b < batch_size && d < dim) {
    float h = h_states[b * dim + d];
    float a = a_seq[d];
    for (int t = 0; t < seq_len; ++t) {
      int idx = (b * seq_len + t) * dim + d;
      h = a * h + b_seq[idx];
      out[idx] = h;
    }
    h_states[b * dim + d] = h;
  }
}

extern "C" void launch_ssm_scan(const float *a_seq, const float *b_seq,
                                float *out, float *h_states, int batch_size,
                                int seq_len, int dim) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(batch_size, (dim + block.x - 1) / block.x);
  ssm_scan_kernel<<<grid, block>>>(a_seq, b_seq, out, h_states, batch_size,
                                   seq_len, dim);
}
