#include "support.h"
#include <math.h>
#include <string.h>

// Timer functions
void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) +
                     (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

// Gaussian kernel generation
GaussianKernel createGaussianKernel(int size, float sigma) {
    GaussianKernel kernel;
    kernel.size = size;
    kernel.sigma = sigma;
    kernel.weights = (float*)malloc(size * sizeof(float));
    
    if (kernel.weights == NULL) {
        FATAL("Unable to allocate Gaussian kernel");
    }
    
    int radius = size / 2;
    float sum = 0.0f;
    
    // Generate 1D Gaussian kernel
    for (int i = 0; i < size; i++) {
        int x = i - radius;
        kernel.weights[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel.weights[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel.weights[i] /= sum;
    }
    
    return kernel;
}

void freeGaussianKernel(GaussianKernel kernel) {
    if (kernel.weights != NULL) {
        free(kernel.weights);
    }
}

// Image allocation and cleanup
Image allocateImage(unsigned int width, unsigned int height, unsigned int channels) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    
    size_t size = width * height * channels * sizeof(float);
    img.data = (float*)malloc(size);
    
    if (img.data == NULL) {
        FATAL("Unable to allocate host image memory");
    }
    
    return img;
}

void freeImage(Image img) {
    if (img.data != NULL) {
        free(img.data);
    }
}

DeviceImage allocateDeviceImage(unsigned int width, unsigned int height, unsigned int channels) {
    DeviceImage img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    
    size_t size = width * height * channels * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&(img.data), size);
    
    if (err != cudaSuccess) {
        FATAL("Unable to allocate device image memory: %s", cudaGetErrorString(err));
    }
    
    return img;
}

void freeDeviceImage(DeviceImage img) {
    if (img.data != NULL) {
        cudaFree(img.data);
    }
}

// Image transfer
void copyImageToDevice(DeviceImage dst, Image src) {
    size_t size = src.width * src.height * src.channels * sizeof(float);
    cudaError_t err = cudaMemcpy(dst.data, src.data, size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        FATAL("Unable to copy image to device: %s", cudaGetErrorString(err));
    }
}

void copyImageFromDevice(Image dst, DeviceImage src) {
    size_t size = src.width * src.height * src.channels * sizeof(float);
    cudaError_t err = cudaMemcpy(dst.data, src.data, size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        FATAL("Unable to copy image from device: %s", cudaGetErrorString(err));
    }
}

// Verification
void verifyBlur(Image reference, Image result, float tolerance) {
    if (reference.width != result.width || 
        reference.height != result.height || 
        reference.channels != result.channels) {
        printf("TEST FAILED: Image dimensions mismatch\n");
        return;
    }
    
    size_t numElements = reference.width * reference.height * reference.channels;
    unsigned int errors = 0;
    float maxError = 0.0f;
    
    for (size_t i = 0; i < numElements; i++) {
        float diff = fabsf(reference.data[i] - result.data[i]);
        if (diff > maxError) {
            maxError = diff;
        }
        if (diff > tolerance) {
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("TEST FAILED: %u/%zu elements exceed tolerance (max error: %.6f)\n", 
               errors, numElements, maxError);
    } else {
        printf("TEST PASSED (max error: %.6f)\n", maxError);
    }
}

float computePSNR(Image img1, Image img2) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        return -1.0f;
    }
    
    size_t numElements = img1.width * img1.height * img1.channels;
    double mse = 0.0;
    
    for (size_t i = 0; i < numElements; i++) {
        double diff = img1.data[i] - img2.data[i];
        mse += diff * diff;
    }
    
    mse /= numElements;
    
    if (mse < 1e-10) {
        return 100.0f; // Images are identical
    }
    
    return 10.0f * log10f(1.0f / mse);
}

// Performance calculation
double calculateGFLOPS(unsigned int width, unsigned int height, int kernelSize, double time) {
    // For 2D convolution: each output pixel requires kernelSize^2 multiplications and additions
    // For separable: each pixel requires 2 * kernelSize operations per pass
    double operations = (double)width * height * kernelSize * 2.0; // Separable approximation
    double gflops = (operations / time) / 1e9;
    return gflops;
}

double calculateBandwidth(size_t bytes, double time) {
    // GB/s
    return (bytes / time) / 1e9;
}
