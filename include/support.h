#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

// Timer structure
typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

// Image structure
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int channels;  // 1 for grayscale, 3 for RGB
    float* data;            // Host data
} Image;

// Device image structure
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int channels;
    float* data;            // Device data
} DeviceImage;

// Gaussian kernel structure
typedef struct {
    float* weights;
    int size;
    float sigma;
} GaussianKernel;

// Performance metrics structure
typedef struct {
    double setupTime;
    double allocTime;
    double h2dTime;
    double kernelTime;
    double d2hTime;
    double freeTime;
    double totalTime;
    double gflops;
    double bandwidth;
} PerfMetrics;

#ifdef __cplusplus
extern "C" {
#endif

// Timer functions
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

// Gaussian kernel generation
GaussianKernel createGaussianKernel(int size, float sigma);
void freeGaussianKernel(GaussianKernel kernel);

// Image allocation and cleanup
Image allocateImage(unsigned int width, unsigned int height, unsigned int channels);
void freeImage(Image img);
DeviceImage allocateDeviceImage(unsigned int width, unsigned int height, unsigned int channels);
void freeDeviceImage(DeviceImage img);

// Image transfer
void copyImageToDevice(DeviceImage dst, Image src);
void copyImageFromDevice(Image dst, DeviceImage src);

// Verification
void verifyBlur(Image reference, Image result, float tolerance);
float computePSNR(Image img1, Image img2);

// Performance calculation
double calculateGFLOPS(unsigned int width, unsigned int height, int kernelSize, double time);
double calculateBandwidth(size_t bytes, double time);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#define CUDA_CHECK(call) \
    do {\
        cudaError_t err = call;\
        if (err != cudaSuccess) {\
            fprintf(stderr, "[%s:%d] CUDA error: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));\
            exit(-1);\
        }\
    } while(0)

#endif // __SUPPORT_H__
