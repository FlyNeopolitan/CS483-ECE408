// Tiled shared memory convolution (2 points)
// Weight matrix (kernel values) in constant memory (1 point)
// Fixed point (FP16) arithmetic. (note this can modify model accuracy slightly) (4 point)
// Tuning with restrict and loop unrolling (considered as one optimization only if you do both) (3 points)

#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"


#define KERNEL_SIZE (7)
#define CHANNEL_SIZE (4)

#define TILE_WIDTH (17)
#define BLOCK_DIM (TILE_WIDTH + KERNEL_SIZE - 1)


__constant__ __half mask[16384]; 

__global__ void conv_forward_kernel(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working
    

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    __shared__ __half tiles[CHANNEL_SIZE * BLOCK_DIM * BLOCK_DIM]; 
    __half* x_half = (__half*)x;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define xh4d(i3, i2, i1, i0) x_half[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define t3d(i2, i1, i0) tiles[(i2) * BLOCK_DIM * BLOCK_DIM + (i1) * BLOCK_DIM + i0]
#define ck4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    // Insert your GPU convolution kernel code here
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + tx;
    __half zero = __float2half(0.0f);
    
    #pragma unroll
    for (int c = 0; c < C; ++c) {
        t3d(c, ty, tx) = (h < H && w < W)? xh4d(b, c, h, w) : zero;
    }
    __syncthreads();
    
    if (ty < TILE_WIDTH && tx < TILE_WIDTH && h < H_out && w < W_out) {
        __half acc = zero;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            #pragma unroll
            for (int p = 0; p < K; ++p) {
                #pragma unroll
                for (int q = 0; q < K; ++q) {
                    __half a = t3d(c, ty + p, tx + q);
                    __half b = ck4d(m, c, p, q);
                    acc = __hfma(a, b, acc);
                }
            }
        }
        y4d(b, m, h, w) = __half2float(acc);
    }
    
#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int y_size = sizeof(float) * H_out * W_out * B * M;
    int x_size = sizeof(__half) * H * W * B * C;
    int k_size = sizeof(__half) * K * K * M * C;

    __half* host_half_k = (__half*)malloc(k_size);
    for (int i = 0; i < K * K * M * C; ++i) {
        host_half_k[i] = __float2half(host_k[i]);
    }
    __half* host_half_x = (__half*)malloc(x_size);
    for (int i = 0; i < H * W * B * C; ++i) {
        host_half_x[i] = __float2half(host_x[i]);
    }
    cudaMalloc(device_y_ptr, y_size);
    cudaMalloc(device_x_ptr, x_size);

    cudaMemcpyToSymbol(mask, host_half_k, k_size);
    cudaMemcpy(*device_x_ptr, host_half_x, x_size, cudaMemcpyHostToDevice);
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    int Y = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH + K - 1, TILE_WIDTH + K - 1, 1);
    dim3 gridDim(M, Y, B);
    conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int y_size = sizeof(float) * H_out * W_out * B * M;
    cudaMemcpy(host_y, device_y, y_size,
                     cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}