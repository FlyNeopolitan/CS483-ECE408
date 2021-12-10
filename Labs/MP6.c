// Histogram Equalization

#include <wb.h>

// The purpose of this lab is to implement an efficient histogramming equalization algorithm for an input image. 
// Like the image convolution MP, the image is represented as RGB float values. 
// You will convert that to GrayScale unsigned char values and compute the histogram. 
// Based on the histogram, you will compute a histogram equalization function which you will then apply to the original image to get the color corrected image.

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here

__global__ void cast_image_float2char(float* input_image, int image_length, unsigned char* uchar_image) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < image_length) {
    uchar_image[idx] = (unsigned char)(255 * input_image[idx]);
  }
}

__global__ void image_rgb2gray(unsigned char* uchar_image, int* histogram, int imageWidth, int imageHeight, unsigned char* gray_image) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int len = imageWidth * imageHeight;
  if (idx < len) {
    float r = uchar_image[3 * idx];
    float g = uchar_image[3 * idx + 1];
    float b = uchar_image[3 * idx + 2];
    gray_image[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
    atomicAdd(&(histogram[gray_image[idx]]), 1);
  }
}

__global__ void scan(int *input, int *output, int len) {
  __shared__ int T[2 * HISTOGRAM_LENGTH];
  // Load data
  int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
  T[threadIdx.x] = (i < len)? input[i] : 0; 
  T[threadIdx.x + blockDim.x] = (i + blockDim.x < len)? input[i + blockDim.x] : 0;
  // Reduction Step
  int stride = 1;
  while (stride < 2 * blockDim.x) {
    __syncthreads();
    int idx = (threadIdx.x + 1) * stride * 2 - 1;
    if (idx < 2 * blockDim.x && idx - stride >= 0) {
      T[idx] += T[idx - stride];
    }
    stride *= 2; 
  }
  // Post Scan Step (Distribution Tree)
  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < 2 * blockDim.x) {
      T[index + stride] += T[index];
    }
    stride = stride / 2;
  }
  // Write back data
  __syncthreads();
  if (i < len) {
    output[i] = T[threadIdx.x];
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = T[threadIdx.x + blockDim.x];
  }
}

__global__ void correct_color(unsigned char* uchar_image, int* cdf, float* outputImage, int image_length, int width, int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int weight = width * height;
  if (idx < image_length) {
    int val = uchar_image[idx];
    float cdfmin = cdf[0] * 1.0 / weight;
    float x = 255 * (cdf[val] * 1.0 / weight - cdfmin) / (1.0 - cdfmin);
    int res = min(max(x, 0.0), 255.0);
    outputImage[idx] = res / 255.0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // 0. Allocate and initialize device memory
  int image_length = imageWidth * imageHeight * imageChannels;
  int size = image_length * sizeof(float);
  float *deviceInputImageData, *deviceOutputImageData;
  cudaMalloc((float**)&deviceInputImageData, size);
  cudaMalloc((float**)&deviceOutputImageData, size);
  cudaMemcpy(deviceInputImageData, hostInputImageData, size, cudaMemcpyHostToDevice);
  // 1. Cast the image from float to unsigned char
  unsigned char* uchar_image;
  cudaMalloc((unsigned char**)&uchar_image, sizeof(unsigned char) * image_length);
  int block_dim = ceil(image_length * 1.0 / (BLOCK_SIZE));
  dim3 gridDim1(block_dim, 1, 1);
  dim3 blockDim1(BLOCK_SIZE, 1, 1);
  cast_image_float2char<<<gridDim1, blockDim1>>>(deviceInputImageData, image_length, uchar_image);
  // 2. Convert the image from RGB to GrayScale, and compute histogram of grayImage
  int image_num_pixels = imageWidth * imageHeight;
  block_dim = ceil(image_num_pixels * 1.0 / (BLOCK_SIZE));
  dim3 gridDim2(block_dim, 1, 1);
  dim3 blockDim2(BLOCK_SIZE, 1, 1);
  int* histogram;
  unsigned char* gray_image;
  cudaMalloc((unsigned int**)&histogram, sizeof(int) * HISTOGRAM_LENGTH);
  cudaMemset(histogram, 0, sizeof(int) * HISTOGRAM_LENGTH);
  cudaMalloc((unsigned char**)&gray_image, sizeof(unsigned char) * image_num_pixels);
  image_rgb2gray<<<gridDim2, blockDim2>>>(uchar_image, histogram, imageWidth, imageHeight, 
                                            gray_image);
  // 3. Compute the Cumulative Distribution Function of histogram
  dim3 gridDim3(1, 1, 1);
  dim3 blockDim3(128, 1, 1);
  int* cdf;
  cudaMalloc((int**)&cdf, sizeof(int) * HISTOGRAM_LENGTH);
  scan<<<gridDim3, blockDim3>>>(histogram, cdf, HISTOGRAM_LENGTH);
  // 4. Apply the histogram equalization function, and Cast back to float
  correct_color<<<gridDim1, blockDim1>>>(uchar_image, cdf, deviceOutputImageData, image_length, imageWidth, imageHeight);
  // 5. Copy result back
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, size, cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);

  //@@ Deallocate
  cudaFree(uchar_image);
  cudaFree(gray_image);
  cudaFree(histogram);
  cudaFree(cdf);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceInputImageData);

  return 0;
}


