// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

// The purpose of this lab is to implement one or more kernels and their associated host code to perform parallel scan on a 1D list. The scan operator used will be addition. You should implement the work- efficient kernel discussed in lecture. Your kernel should be able to handle input lists of arbitrary length. However, for simplicity, you can assume that the input list will be at most 2,048 * 2,048 elements.

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void load_auxiliary(float *output, float* auxiliary, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 2 * BLOCK_SIZE - 1 < len && 0 <= i * 2 * BLOCK_SIZE - 1) {
    auxiliary[i] = output[i * 2 * BLOCK_SIZE - 1];
  } else {
    auxiliary[i] = 0;
  }
  __syncthreads();
}

__global__ void add(float* deviceOutput, float* device_auxiliaryOutput, int len) {
  int i = blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x;
  if (i < len) {
    deviceOutput[i] += device_auxiliaryOutput[blockIdx.x];
  }
  if (i + BLOCK_SIZE < len) {
    deviceOutput[i + BLOCK_SIZE] += device_auxiliaryOutput[blockIdx.x];
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  // Load data
  int i = blockIdx.x * 2 * BLOCK_SIZE + threadIdx.x;
  T[threadIdx.x] = (i < len)? input[i] : 0; 
  T[threadIdx.x + BLOCK_SIZE] = (i + BLOCK_SIZE < len)? input[i + BLOCK_SIZE] : 0;
  // Reduction Step
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int idx = (threadIdx.x + 1) * stride * 2 - 1;
    if (idx < 2 * BLOCK_SIZE && idx - stride >= 0) {
      T[idx] += T[idx - stride];
    }
    stride *= 2; 
  }
  // Post Scan Step (Distribution Tree)
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
    stride = stride / 2;
  }
  // Write back data
  __syncthreads();
  if (i < len) {
    output[i] = T[threadIdx.x];
  }
  if (i + BLOCK_SIZE < len) {
    output[i + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int block_dim = ceil(numElements / (2.0 * BLOCK_SIZE));
  dim3 gridDim (block_dim, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  // 1. scan initial array 
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);
  // 2. store block sums to auxlilary Array
  float *device_auxiliaryInput, *device_auxiliaryOutput;
  cudaMalloc((void **)&device_auxiliaryInput, block_dim * sizeof(float));
  load_auxiliary<<<dim3(block_dim, 1, 1), dim3(1, 1, 1) >>>(deviceOutput, device_auxiliaryInput, numElements);
  // 3. scan the auxiliary Array
  cudaMalloc((void **)&device_auxiliaryOutput, block_dim * sizeof(float));
  scan<<<dim3(ceil(block_dim / (2.0 * BLOCK_SIZE)),1,1), blockDim>>>(device_auxiliaryInput, device_auxiliaryOutput, block_dim);
  // 4. add the auxiliary Array to result
  add<<<gridDim, blockDim>>>(deviceOutput, device_auxiliaryOutput, numElements);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(device_auxiliaryInput); // free allocated memory
  cudaFree(device_auxiliaryOutput); // free allocated memory
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

