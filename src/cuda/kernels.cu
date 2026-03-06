#ifdef __INTELLISENSE__
#define __global__
#include <device_functions.h>
#include <device_launch_parameters.h>
#endif

extern "C" __global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < 0.0f) data[idx] = 0.0f;
    }
}

extern "C" __global__ void add_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] += b[idx];
}
