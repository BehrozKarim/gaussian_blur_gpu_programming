#include "../include/support.h"
#include "../include/image_io.h"
#include "../include/kernels.cuh"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// External CPU functions
extern "C" void cpuBlurSeparable(Image input, Image output, GaussianKernel kernel);
extern "C" void cpuBlurBatch(Image* inputs, Image* outputs, unsigned int count, GaussianKernel kernel);

// ============================================================================
// GPU Wrapper Functions
// ============================================================================

void runNaiveBlur(DeviceImage d_input, DeviceImage d_output, float* d_kernel2D, 
                  int kernelSize, PerfMetrics* metrics) {
    dim3 blockDim(16, 16);
    dim3 gridDim((d_input.width + blockDim.x - 1) / blockDim.x,
                 (d_input.height + blockDim.y - 1) / blockDim.y);
    
    Timer timer;
    startTime(&timer);
    
    blurNaive<<<gridDim, blockDim>>>(d_input.data, d_output.data,
                                     d_input.width, d_input.height, d_input.channels,
                                     d_kernel2D, kernelSize);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    stopTime(&timer);
    metrics->kernelTime = elapsedTime(timer);
}

void runTiledBlur(DeviceImage d_input, DeviceImage d_output, float* d_kernel2D,
                  int kernelSize, PerfMetrics* metrics) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((d_input.width + TILE_SIZE - 1) / TILE_SIZE,
                 (d_input.height + TILE_SIZE - 1) / TILE_SIZE);
    
    Timer timer;
    startTime(&timer);
    
    blurTiled<<<gridDim, blockDim>>>(d_input.data, d_output.data,
                                     d_input.width, d_input.height, d_input.channels,
                                     d_kernel2D, kernelSize);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    stopTime(&timer);
    metrics->kernelTime = elapsedTime(timer);
}

void runSeparableBlur(DeviceImage d_input, DeviceImage d_temp, DeviceImage d_output,
                      float* d_kernel1D, int kernelSize, PerfMetrics* metrics) {
    dim3 blockDim(16, 16);
    dim3 gridDim((d_input.width + blockDim.x - 1) / blockDim.x,
                 (d_input.height + blockDim.y - 1) / blockDim.y);
    
    Timer timer;
    startTime(&timer);
    
    // Horizontal pass
    blurSeparableHorizontal<<<gridDim, blockDim>>>(d_input.data, d_temp.data,
                                                    d_input.width, d_input.height, d_input.channels,
                                                    d_kernel1D, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Vertical pass
    blurSeparableVertical<<<gridDim, blockDim>>>(d_temp.data, d_output.data,
                                                  d_input.width, d_input.height, d_input.channels,
                                                  d_kernel1D, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    stopTime(&timer);
    metrics->kernelTime = elapsedTime(timer);
}

void runSeparableConstantBlur(DeviceImage d_input, DeviceImage d_temp, DeviceImage d_output,
                              int kernelSize, PerfMetrics* metrics) {
    dim3 blockDim(16, 16);
    dim3 gridDim((d_input.width + blockDim.x - 1) / blockDim.x,
                 (d_input.height + blockDim.y - 1) / blockDim.y);
    
    Timer timer;
    startTime(&timer);
    
    // Horizontal pass
    blurSeparableConstantHorizontal<<<gridDim, blockDim>>>(d_input.data, d_temp.data,
                                                            d_input.width, d_input.height, d_input.channels,
                                                            kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Vertical pass
    blurSeparableConstantVertical<<<gridDim, blockDim>>>(d_temp.data, d_output.data,
                                                          d_input.width, d_input.height, d_input.channels,
                                                          kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    stopTime(&timer);
    metrics->kernelTime = elapsedTime(timer);
}

void runSeparableSharedBlur(DeviceImage d_input, DeviceImage d_temp, DeviceImage d_output,
                            int kernelSize, PerfMetrics* metrics) {
    dim3 blockDimH(SHARED_TILE_WIDTH, 1);
    dim3 gridDimH((d_input.width + blockDimH.x - 1) / blockDimH.x, d_input.height);
    
    dim3 blockDimV(SHARED_TILE_WIDTH, SHARED_TILE_HEIGHT);
    dim3 gridDimV((d_input.width + blockDimV.x - 1) / blockDimV.x,
                  (d_input.height + blockDimV.y - 1) / blockDimV.y);
    
    Timer timer;
    startTime(&timer);
    
    // Horizontal pass
    blurSeparableSharedHorizontal<<<gridDimH, blockDimH>>>(d_input.data, d_temp.data,
                                                            d_input.width, d_input.height, d_input.channels,
                                                            kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Vertical pass
    blurSeparableSharedVertical<<<gridDimV, blockDimV>>>(d_temp.data, d_output.data,
                                                          d_input.width, d_input.height, d_input.channels,
                                                          kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    stopTime(&timer);
    metrics->kernelTime = elapsedTime(timer);
}

// ============================================================================
// Batch Processing with CUDA Streams
// ============================================================================

void runBatchBlurWithStreams(Image* h_inputs, Image* h_outputs, unsigned int batchSize,
                             GaussianKernel kernel, const char* method) {
    printf("\n=== Batch Processing: %u images using %s ===\n", batchSize, method);
    
    int kernelSize = kernel.size;
    const int numStreams = 4;
    cudaStream_t streams[numStreams];
    
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate device memory for all images
    DeviceImage* d_inputs = (DeviceImage*)malloc(batchSize * sizeof(DeviceImage));
    DeviceImage* d_temps = (DeviceImage*)malloc(batchSize * sizeof(DeviceImage));
    DeviceImage* d_outputs = (DeviceImage*)malloc(batchSize * sizeof(DeviceImage));
    
    for (unsigned int i = 0; i < batchSize; i++) {
        d_inputs[i] = allocateDeviceImage(h_inputs[i].width, h_inputs[i].height, h_inputs[i].channels);
        d_temps[i] = allocateDeviceImage(h_inputs[i].width, h_inputs[i].height, h_inputs[i].channels);
        d_outputs[i] = allocateDeviceImage(h_inputs[i].width, h_inputs[i].height, h_inputs[i].channels);
    }
    
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian1D, kernel.weights, kernelSize * sizeof(float)));
    
    Timer totalTimer;
    startTime(&totalTimer);
    
    // Process images in streams
    for (unsigned int i = 0; i < batchSize; i++) {
        int streamId = i % numStreams;
        
        // Async copy H2D
        size_t imageSize = h_inputs[i].width * h_inputs[i].height * h_inputs[i].channels * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_inputs[i].data, h_inputs[i].data, imageSize, 
                                   cudaMemcpyHostToDevice, streams[streamId]));
        
        // Launch kernels
        dim3 blockDim(16, 16);
        dim3 gridDim((h_inputs[i].width + blockDim.x - 1) / blockDim.x,
                     (h_inputs[i].height + blockDim.y - 1) / blockDim.y);
        
        blurSeparableConstantHorizontal<<<gridDim, blockDim, 0, streams[streamId]>>>(
            d_inputs[i].data, d_temps[i].data,
            d_inputs[i].width, d_inputs[i].height, d_inputs[i].channels, kernelSize);
        
        blurSeparableConstantVertical<<<gridDim, blockDim, 0, streams[streamId]>>>(
            d_temps[i].data, d_outputs[i].data,
            d_inputs[i].width, d_inputs[i].height, d_inputs[i].channels, kernelSize);
        
        // Async copy D2H
        CUDA_CHECK(cudaMemcpyAsync(h_outputs[i].data, d_outputs[i].data, imageSize,
                                   cudaMemcpyDeviceToHost, streams[streamId]));
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    stopTime(&totalTimer);
    double totalTime = elapsedTime(totalTimer);
    
    printf("Total batch processing time: %.6f s\n", totalTime);
    printf("Throughput: %.2f images/sec\n", batchSize / totalTime);
    printf("Average time per image: %.6f s\n", totalTime / batchSize);
    
    // Cleanup
    for (unsigned int i = 0; i < batchSize; i++) {
        freeDeviceImage(d_inputs[i]);
        freeDeviceImage(d_temps[i]);
        freeDeviceImage(d_outputs[i]);
    }
    free(d_inputs);
    free(d_temps);
    free(d_outputs);
    
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

// ============================================================================
// Main Performance Comparison
// ============================================================================

void printHeader() {
    printf("\n");
    printf("================================================================================\n");
    printf("                   GPU GAUSSIAN BLUR - PERFORMANCE BENCHMARK                   \n");
    printf("================================================================================\n");
}

void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("\nGPU Device Information:\n");
    printf("  Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Number of SMs: %d\n", prop.multiProcessorCount);
    printf("\n");
}

void runComparison(Image input, GaussianKernel kernel) {
    printf("\nImage: %ux%u, %d channels\n", input.width, input.height, input.channels);
    printf("Gaussian kernel: size=%d, sigma=%.2f\n", kernel.size, kernel.sigma);
    printf("--------------------------------------------------------------------------------\n");
    
    int kernelSize = kernel.size;
    
    // Prepare outputs
    Image output_cpu = allocateImage(input.width, input.height, input.channels);
    Image output_gpu = allocateImage(input.width, input.height, input.channels);
    
    // ========== CPU BASELINE ==========
    printf("\n[1/4] CPU Baseline (OpenMP)...\n");
    Timer cpuTimer;
    startTime(&cpuTimer);
    cpuBlurSeparable(input, output_cpu, kernel);
    stopTime(&cpuTimer);
    double cpuTime = elapsedTime(cpuTimer);
    printf("  Time: %.6f s\n", cpuTime);
    
    // Prepare GPU memory
    DeviceImage d_input = allocateDeviceImage(input.width, input.height, input.channels);
    DeviceImage d_temp = allocateDeviceImage(input.width, input.height, input.channels);
    DeviceImage d_output = allocateDeviceImage(input.width, input.height, input.channels);
    
    // Prepare kernel data
    float* kernel2D = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel2D[i * kernelSize + j] = kernel.weights[i] * kernel.weights[j];
        }
    }
    
    float *d_kernel1D, *d_kernel2D;
    CUDA_CHECK(cudaMalloc(&d_kernel1D, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel2D, kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kernel1D, kernel.weights, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel2D, kernel2D, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy kernel to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian1D, kernel.weights, kernelSize * sizeof(float)));
    
    // Copy input to device
    copyImageToDevice(d_input, input);
    
    PerfMetrics metrics;
    
    // ========== NAIVE ==========
    printf("\n[2/4] Naive (Global Memory)...\n");
    runNaiveBlur(d_input, d_output, d_kernel2D, kernelSize, &metrics);
    copyImageFromDevice(output_gpu, d_output);
    printf("  Kernel time: %.6f s\n", metrics.kernelTime);
    printf("  Speedup vs CPU: %.2fx\n", cpuTime / metrics.kernelTime);
    verifyBlur(output_cpu, output_gpu, 0.001f);
    
    // ========== TILED ==========
    printf("\n[3/4] Tiled (Shared Memory)...\n");
    runTiledBlur(d_input, d_output, d_kernel2D, kernelSize, &metrics);
    copyImageFromDevice(output_gpu, d_output);
    printf("  Kernel time: %.6f s\n", metrics.kernelTime);
    printf("  Speedup vs CPU: %.2fx\n", cpuTime / metrics.kernelTime);
    verifyBlur(output_cpu, output_gpu, 0.001f);
    
    // ========== SEPARABLE ==========
    printf("\n[4/4] Separable (Global Memory - Best)...\n");
    runSeparableBlur(d_input, d_temp, d_output, d_kernel1D, kernelSize, &metrics);
    copyImageFromDevice(output_gpu, d_output);
    printf("  Kernel time: %.6f s\n", metrics.kernelTime);
    printf("  Speedup vs CPU: %.2fx\n", cpuTime / metrics.kernelTime);
    verifyBlur(output_cpu, output_gpu, 0.001f);
    
    // Cleanup
    freeImage(output_cpu);
    freeImage(output_gpu);
    freeDeviceImage(d_input);
    freeDeviceImage(d_temp);
    freeDeviceImage(d_output);
    cudaFree(d_kernel1D);
    cudaFree(d_kernel2D);
    free(kernel2D);
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    printHeader();
    printDeviceInfo();
    
    // Parse command line arguments
    unsigned int width = 1024;
    unsigned int height = 1024;
    unsigned int channels = 3;
    int kernelSize = 7;
    float sigma = 2.0f;
    unsigned int batchSize = 10;
    const char* inputImagePath = NULL;
    bool batchMode = false;
    char** imagePaths = NULL;
    int imageCount = 0;
    
    if (argc >= 2) {
        // Check for batch mode flag
        if (strcmp(argv[1], "--batch") == 0 && argc >= 4) {
            batchMode = true;
            kernelSize = atoi(argv[2]);
            sigma = atof(argv[3]);
            // Collect all image paths from argv[4] onwards
            imageCount = argc - 4;
            imagePaths = &argv[4];
        }
        // Check if first argument is an image file (contains . for extension)
        else if (strchr(argv[1], '.') != NULL) {
            inputImagePath = argv[1];
            if (argc >= 3) kernelSize = atoi(argv[2]);
            if (argc >= 4) sigma = atof(argv[3]);
            if (argc >= 5) batchSize = atoi(argv[4]);
        } else {
            width = atoi(argv[1]);
            if (argc >= 3) height = atoi(argv[2]);
            if (argc >= 4) kernelSize = atoi(argv[3]);
            if (argc >= 5) sigma = atof(argv[4]);
            if (argc >= 6) batchSize = atoi(argv[5]);
        }
    }
    
    // Handle batch mode for multiple real images
    if (batchMode && imageCount > 0) {
        printf("\n========================================\n");
        printf("BATCH MODE: Processing %d images\n", imageCount);
        printf("Kernel size: %d, Sigma: %.2f\n", kernelSize, sigma);
        printf("========================================\n\n");
        
        // Load all images
        Image* images = (Image*)malloc(imageCount * sizeof(Image));
        for (int i = 0; i < imageCount; i++) {
            printf("Loading image %d/%d: %s\n", i+1, imageCount, imagePaths[i]);
            images[i] = loadImage(imagePaths[i]);
            if (images[i].data == NULL) {
                fprintf(stderr, "Failed to load image: %s\n", imagePaths[i]);
                continue;
            }
        }
        
        // Create Gaussian kernel
        GaussianKernel kernel = createGaussianKernel(kernelSize, sigma);
        
        // Process all images using batch GPU processing
        printf("\nProcessing batch with CUDA streams...\n");
        cudaStream_t* streams = (cudaStream_t*)malloc(4 * sizeof(cudaStream_t));
        for (int i = 0; i < 4; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        
        DeviceImage* d_inputs = (DeviceImage*)malloc(imageCount * sizeof(DeviceImage));
        DeviceImage* d_temps = (DeviceImage*)malloc(imageCount * sizeof(DeviceImage));
        DeviceImage* d_outputs = (DeviceImage*)malloc(imageCount * sizeof(DeviceImage));
        
        for (int i = 0; i < imageCount; i++) {
            d_inputs[i] = allocateDeviceImage(images[i].width, images[i].height, images[i].channels);
            d_temps[i] = allocateDeviceImage(images[i].width, images[i].height, images[i].channels);
            d_outputs[i] = allocateDeviceImage(images[i].width, images[i].height, images[i].channels);
        }
        
        // Copy kernel to device
        float* d_kernel1D;
        CUDA_CHECK(cudaMalloc(&d_kernel1D, kernelSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel1D, kernel.weights, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian1D, kernel.weights, kernelSize * sizeof(float)));
        
        Timer totalTimer;
        startTime(&totalTimer);
        
        // Process images in streams
        for (int i = 0; i < imageCount; i++) {
            int streamId = i % 4;
            
            // Async copy H2D
            size_t imageSize = images[i].width * images[i].height * images[i].channels * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(d_inputs[i].data, images[i].data, imageSize, 
                                       cudaMemcpyHostToDevice, streams[streamId]));
            
            // Launch kernels
            dim3 blockSize(16, 16);
            dim3 gridSize((images[i].width + blockSize.x - 1) / blockSize.x,
                         (images[i].height + blockSize.y - 1) / blockSize.y);
            
            blurSeparableHorizontal<<<gridSize, blockSize, 0, streams[streamId]>>>(
                d_inputs[i].data, d_temps[i].data,
                images[i].width, images[i].height, images[i].channels,
                d_kernel1D, kernelSize);
            
            blurSeparableVertical<<<gridSize, blockSize, 0, streams[streamId]>>>(
                d_temps[i].data, d_outputs[i].data,
                images[i].width, images[i].height, images[i].channels,
                d_kernel1D, kernelSize);
            
            // Async copy D2H
            CUDA_CHECK(cudaMemcpyAsync(images[i].data, d_outputs[i].data, imageSize,
                                       cudaMemcpyDeviceToHost, streams[streamId]));
        }
        
        // Wait for all streams
        for (int i = 0; i < 4; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        stopTime(&totalTimer);
        double totalTime = elapsedTime(totalTimer);
        
        printf("Batch processing complete!\n");
        printf("  Total time: %.6f s\n", totalTime);
        printf("  Throughput: %.2f images/sec\n", imageCount / totalTime);
        printf("  Average time per image: %.6f s\n", totalTime / imageCount);
        printf("\nSaving output images...\n");
        
        system("mkdir -p output_images");
        
        // Save all output images
        for (int i = 0; i < imageCount; i++) {
            char outputPath[512];
            const char* inputFilename = strrchr(imagePaths[i], '/');
            if (inputFilename == NULL) inputFilename = strrchr(imagePaths[i], '\\');
            if (inputFilename == NULL) inputFilename = imagePaths[i];
            else inputFilename++;
            
            char baseFilename[256];
            strncpy(baseFilename, inputFilename, sizeof(baseFilename) - 1);
            char* dot = strrchr(baseFilename, '.');
            if (dot != NULL) *dot = '\0';
            
            snprintf(outputPath, sizeof(outputPath), 
                     "output_images/%s_blur_k%d_s%.1f.png", 
                     baseFilename, kernelSize, sigma);
            
            saveImage(outputPath, images[i]);
            printf("  Saved: %s\n", outputPath);
        }
        
        // Cleanup
        for (int i = 0; i < imageCount; i++) {
            freeImage(images[i]);
            freeDeviceImage(d_inputs[i]);
            freeDeviceImage(d_temps[i]);
            freeDeviceImage(d_outputs[i]);
        }
        free(images);
        free(d_inputs);
        free(d_temps);
        free(d_outputs);
        cudaFree(d_kernel1D);
        for (int i = 0; i < 4; i++) {
            cudaStreamDestroy(streams[i]);
        }
        free(streams);
        freeGaussianKernel(kernel);
        
        printf("\n========================================\n");
        printf("Batch processing completed successfully!\n");
        printf("========================================\n");
        
        return 0;
    }
    
    // Load or generate image (single image mode)
    Image testImage;
    if (inputImagePath != NULL) {
        printf("\nLoading image from: %s\n", inputImagePath);
        testImage = loadImage(inputImagePath);
        width = testImage.width;
        height = testImage.height;
        channels = testImage.channels;
    } else {
        printf("\nGenerating synthetic test image...\n");
        testImage = generateTestImage(width, height, channels);
    }
    
    printf("\nConfiguration:\n");
    printf("  Image size: %ux%u\n", width, height);
    printf("  Channels: %d\n", channels);
    printf("  Kernel size: %d\n", kernelSize);
    printf("  Sigma: %.2f\n", sigma);
    printf("  Batch size: %u\n", batchSize);
    
    // Create Gaussian kernel
    GaussianKernel kernel = createGaussianKernel(kernelSize, sigma);
    
    // Run single image comparison
    runComparison(testImage, kernel);
    
    // Save output if processing a real image
    if (inputImagePath != NULL) {
        // Generate unique output filename based on input and parameters
        char outputPath[512];
        const char* inputFilename = strrchr(inputImagePath, '/');
        if (inputFilename == NULL) inputFilename = strrchr(inputImagePath, '\\');
        if (inputFilename == NULL) inputFilename = inputImagePath;
        else inputFilename++; // Skip the slash
        
        // Remove extension from input filename
        char baseFilename[256];
        strncpy(baseFilename, inputFilename, sizeof(baseFilename) - 1);
        char* dot = strrchr(baseFilename, '.');
        if (dot != NULL) *dot = '\0';
        
        snprintf(outputPath, sizeof(outputPath), 
                 "output_images/%s_blur_k%d_s%.1f.png", 
                 baseFilename, kernelSize, sigma);
        
        printf("\nSaving output image to %s...\n", outputPath);
        system("mkdir -p output_images");
        
        // Re-run best performing kernel to get output
        DeviceImage d_input = allocateDeviceImage(width, height, channels);
        DeviceImage d_temp = allocateDeviceImage(width, height, channels);
        DeviceImage d_output = allocateDeviceImage(width, height, channels);
        float* d_kernel1D;
        CUDA_CHECK(cudaMalloc(&d_kernel1D, kernelSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel1D, kernel.weights, kernelSize * sizeof(float), cudaMemcpyHostToDevice));
        
        copyImageToDevice(d_input, testImage);
        PerfMetrics metrics;
        runSeparableBlur(d_input, d_temp, d_output, d_kernel1D, kernelSize, &metrics);
        
        Image output = allocateImage(width, height, channels);
        copyImageFromDevice(output, d_output);
        saveImage(outputPath, output);
        
        freeImage(output);
        freeDeviceImage(d_input);
        freeDeviceImage(d_temp);
        freeDeviceImage(d_output);
        cudaFree(d_kernel1D);
        printf("Output saved successfully!\n");
    }
    
    // Run batch processing
    printf("\n\n================================================================================\n");
    printf("                           BATCH PROCESSING TEST                            \n");
    printf("================================================================================\n");
    
    Image* batchInputs = generateTestBatch(batchSize, width, height, channels);
    Image* batchOutputs = (Image*)malloc(batchSize * sizeof(Image));
    for (unsigned int i = 0; i < batchSize; i++) {
        batchOutputs[i] = allocateImage(width, height, channels);
    }
    
    // CPU batch
    printf("\nCPU Batch Processing...\n");
    Timer batchTimer;
    startTime(&batchTimer);
    cpuBlurBatch(batchInputs, batchOutputs, batchSize, kernel);
    stopTime(&batchTimer);
    double cpuBatchTime = elapsedTime(batchTimer);
    printf("  Total time: %.6f s\n", cpuBatchTime);
    printf("  Throughput: %.2f images/sec\n", batchSize / cpuBatchTime);
    
    // GPU batch with streams
    runBatchBlurWithStreams(batchInputs, batchOutputs, batchSize, kernel, "CUDA Streams");
    
    // Cleanup
    freeBatch(batchInputs, batchSize);
    for (unsigned int i = 0; i < batchSize; i++) {
        freeImage(batchOutputs[i]);
    }
    free(batchOutputs);
    freeImage(testImage);
    freeGaussianKernel(kernel);
    
    printf("\n================================================================================\n");
    printf("                         BENCHMARK COMPLETED                                \n");
    printf("================================================================================\n\n");
    
    return 0;
}
