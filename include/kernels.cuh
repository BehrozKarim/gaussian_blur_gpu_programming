#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "support.h"

// Maximum filter size for constant memory
#define MAX_FILTER_SIZE 15
#define MAX_FILTER_ELEMENTS (MAX_FILTER_SIZE * MAX_FILTER_SIZE)

// Tile sizes for different kernels
#define TILE_SIZE 16          // For tiled blur kernel
#define SHARED_TILE_WIDTH 128 // For separable shared memory kernel
#define SHARED_TILE_HEIGHT 16 // For separable shared memory kernel

// Constant memory for Gaussian kernel
__constant__ float d_gaussianKernel[MAX_FILTER_ELEMENTS];
__constant__ float d_gaussian1D[MAX_FILTER_SIZE];

// Kernel 1: Naive 2D Convolution (Global Memory Only)
__global__ void blurNaive(float* input, float* output, 
                          unsigned int width, unsigned int height, unsigned int channels,
                          float* kernel, int kernelSize);

// Kernel 2: Tiled 2D Convolution (Shared Memory)
__global__ void blurTiled(float* input, float* output,
                          unsigned int width, unsigned int height, unsigned int channels,
                          float* kernel, int kernelSize);

// Kernel 3: Separable 1D Horizontal Pass
__global__ void blurSeparableHorizontal(float* input, float* output,
                                        unsigned int width, unsigned int height, unsigned int channels,
                                        float* kernel1D, int kernelSize);

// Kernel 4: Separable 1D Vertical Pass
__global__ void blurSeparableVertical(float* input, float* output,
                                      unsigned int width, unsigned int height, unsigned int channels,
                                      float* kernel1D, int kernelSize);

// Kernel 5: Separable with Constant Memory (Horizontal)
__global__ void blurSeparableConstantHorizontal(float* input, float* output,
                                                unsigned int width, unsigned int height, unsigned int channels,
                                                int kernelSize);

// Kernel 6: Separable with Constant Memory (Vertical)
__global__ void blurSeparableConstantVertical(float* input, float* output,
                                              unsigned int width, unsigned int height, unsigned int channels,
                                              int kernelSize);

// Kernel 7: Separable with Shared Memory (Horizontal)
__global__ void blurSeparableSharedHorizontal(float* input, float* output,
                                              unsigned int width, unsigned int height, unsigned int channels,
                                              int kernelSize);

// Kernel 8: Separable with Shared Memory (Vertical)
__global__ void blurSeparableSharedVertical(float* input, float* output,
                                            unsigned int width, unsigned int height, unsigned int channels,
                                            int kernelSize);

#endif // __KERNELS_CUH__
