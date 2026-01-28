#include "../include/kernels.cuh"
#include <stdio.h>

// ============================================================================
// Kernel 1: Naive 2D Convolution (Global Memory Only)
// ============================================================================
__global__ void blurNaive(float* input, float* output, 
                          unsigned int width, unsigned int height, unsigned int channels,
                          float* kernel, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int radius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int y = row + ky - radius;
                int x = col + kx - radius;
                
                // Clamp to image boundaries
                y = max(0, min(y, (int)height - 1));
                x = max(0, min(x, (int)width - 1));
                
                float kernelVal = kernel[ky * kernelSize + kx];
                float pixelVal = input[(y * width + x) * channels + c];
                sum += kernelVal * pixelVal;
            }
        }
        
        output[(row * width + col) * channels + c] = sum;
    }
}

// ============================================================================
// Kernel 2: Tiled 2D Convolution (Shared Memory)
// ============================================================================
#define TILE_SIZE 16
#define BLOCK_SIZE_TILED (TILE_SIZE + 8) // Assuming max kernel radius of 4

__global__ void blurTiled(float* input, float* output,
                          unsigned int width, unsigned int height, unsigned int channels,
                          float* kernel, int kernelSize) {
    __shared__ float tile[BLOCK_SIZE_TILED][BLOCK_SIZE_TILED];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    int radius = kernelSize / 2;
    
    // Process each channel separately
    for (int c = 0; c < channels; c++) {
        // Load tile with halo (each thread may load multiple pixels)
        for (int dy = ty; dy < BLOCK_SIZE_TILED; dy += blockDim.y) {
            for (int dx = tx; dx < BLOCK_SIZE_TILED; dx += blockDim.x) {
                int imgRow = blockIdx.y * TILE_SIZE + dy - radius;
                int imgCol = blockIdx.x * TILE_SIZE + dx - radius;
                
                // Clamp to boundaries
                imgRow = max(0, min(imgRow, (int)height - 1));
                imgCol = max(0, min(imgCol, (int)width - 1));
                
                tile[dy][dx] = input[(imgRow * width + imgCol) * channels + c];
            }
        }
        
        __syncthreads();
        
        // Compute convolution
        if (row < height && col < width && tx < TILE_SIZE && ty < TILE_SIZE) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int tileY = ty + ky;
                    int tileX = tx + kx;
                    sum += kernel[ky * kernelSize + kx] * tile[tileY][tileX];
                }
            }
            
            output[(row * width + col) * channels + c] = sum;
        }
        
        __syncthreads();
    }
}

// ============================================================================
// Kernel 3 & 4: Separable Convolution (Global Memory)
// ============================================================================
__global__ void blurSeparableHorizontal(float* input, float* output,
                                        unsigned int width, unsigned int height, unsigned int channels,
                                        float* kernel1D, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int radius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernelSize; k++) {
            int x = col + k - radius;
            x = max(0, min(x, (int)width - 1));
            
            sum += kernel1D[k] * input[(row * width + x) * channels + c];
        }
        
        output[(row * width + col) * channels + c] = sum;
    }
}

__global__ void blurSeparableVertical(float* input, float* output,
                                      unsigned int width, unsigned int height, unsigned int channels,
                                      float* kernel1D, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int radius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernelSize; k++) {
            int y = row + k - radius;
            y = max(0, min(y, (int)height - 1));
            
            sum += kernel1D[k] * input[(y * width + col) * channels + c];
        }
        
        output[(row * width + col) * channels + c] = sum;
    }
}

// ============================================================================
// Kernel 5 & 6: Separable with Constant Memory
// ============================================================================
__global__ void blurSeparableConstantHorizontal(float* input, float* output,
                                                unsigned int width, unsigned int height, unsigned int channels,
                                                int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int radius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernelSize; k++) {
            int x = col + k - radius;
            x = max(0, min(x, (int)width - 1));
            
            sum += d_gaussian1D[k] * input[(row * width + x) * channels + c];
        }
        
        output[(row * width + col) * channels + c] = sum;
    }
}

__global__ void blurSeparableConstantVertical(float* input, float* output,
                                              unsigned int width, unsigned int height, unsigned int channels,
                                              int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int radius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = 0; k < kernelSize; k++) {
            int y = row + k - radius;
            y = max(0, min(y, (int)height - 1));
            
            sum += d_gaussian1D[k] * input[(y * width + col) * channels + c];
        }
        
        output[(row * width + col) * channels + c] = sum;
    }
}

// ============================================================================
// Kernel 7 & 8: Separable with Shared Memory
// ============================================================================
#define SHARED_TILE_WIDTH 128
#define SHARED_TILE_HEIGHT 16

__global__ void blurSeparableSharedHorizontal(float* input, float* output,
                                              unsigned int width, unsigned int height, unsigned int channels,
                                              int kernelSize) {
    __shared__ float sharedRow[SHARED_TILE_WIDTH + MAX_FILTER_SIZE];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int radius = kernelSize / 2;
    
    if (row >= height) return;
    
    for (int c = 0; c < channels; c++) {
        // Load main data into shared memory (with halo)
        int loadCol = blockIdx.x * blockDim.x + tx - radius;
        loadCol = max(0, min(loadCol, (int)width - 1));
        sharedRow[tx + radius] = input[(row * width + loadCol) * channels + c];
        
        // Load left halo
        if (tx < radius) {
            int leftCol = blockIdx.x * blockDim.x - radius + tx;
            leftCol = max(0, min(leftCol, (int)width - 1));
            sharedRow[tx] = input[(row * width + leftCol) * channels + c];
        }
        
        // Load right halo
        if (tx < radius) {
            int rightCol = blockIdx.x * blockDim.x + blockDim.x + tx;
            rightCol = min(rightCol, (int)width - 1);
            sharedRow[blockDim.x + radius + tx] = input[(row * width + rightCol) * channels + c];
        }
        
        __syncthreads();
        
        if (col < width) {
            float sum = 0.0f;
            for (int k = 0; k < kernelSize; k++) {
                sum += d_gaussian1D[k] * sharedRow[tx + radius + k - radius];
            }
            output[(row * width + col) * channels + c] = sum;
        }
        
        __syncthreads();
    }
}

__global__ void blurSeparableSharedVertical(float* input, float* output,
                                            unsigned int width, unsigned int height, unsigned int channels,
                                            int kernelSize) {
    __shared__ float sharedCol[SHARED_TILE_HEIGHT + MAX_FILTER_SIZE][SHARED_TILE_WIDTH];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int radius = kernelSize / 2;
    
    if (col >= width) return;
    
    for (int c = 0; c < channels; c++) {
        // Load main data into shared memory
        int loadRow = blockIdx.y * blockDim.y + ty - radius;
        loadRow = max(0, min(loadRow, (int)height - 1));
        sharedCol[ty + radius][tx] = input[(loadRow * width + col) * channels + c];
        
        // Load top halo
        if (ty < radius) {
            int topRow = blockIdx.y * blockDim.y - radius + ty;
            topRow = max(0, min(topRow, (int)height - 1));
            sharedCol[ty][tx] = input[(topRow * width + col) * channels + c];
        }
        
        // Load bottom halo
        if (ty < radius) {
            int bottomRow = blockIdx.y * blockDim.y + blockDim.y + ty;
            bottomRow = min(bottomRow, (int)height - 1);
            sharedCol[blockDim.y + radius + ty][tx] = input[(bottomRow * width + col) * channels + c];
        }
        
        __syncthreads();
        
        if (row < height) {
            float sum = 0.0f;
            for (int k = 0; k < kernelSize; k++) {
                sum += d_gaussian1D[k] * sharedCol[ty + radius + k - radius][tx];
            }
            output[(row * width + col) * channels + c] = sum;
        }
        
        __syncthreads();
    }
}
