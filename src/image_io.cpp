#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include "../include/image_io.h"
#include <string.h>
#include <math.h>

Image loadImage(const char* filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    
    if (data == NULL) {
        fprintf(stderr, "Error loading image: %s\n", filename);
        FATAL("Failed to load image");
    }
    
    Image img = allocateImage(width, height, channels);
    
    // Convert to float [0, 1]
    size_t numElements = width * height * channels;
    for (size_t i = 0; i < numElements; i++) {
        img.data[i] = data[i] / 255.0f;
    }
    
    stbi_image_free(data);
    
    printf("Loaded image: %s (%dx%d, %d channels)\n", filename, width, height, channels);
    return img;
}

void saveImage(const char* filename, Image img) {
    // Convert float to unsigned char
    size_t numElements = img.width * img.height * img.channels;
    unsigned char* data = (unsigned char*)malloc(numElements);
    
    if (data == NULL) {
        FATAL("Unable to allocate memory for image saving");
    }
    
    for (size_t i = 0; i < numElements; i++) {
        float val = img.data[i] * 255.0f;
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        data[i] = (unsigned char)val;
    }
    
    // Determine format from extension
    const char* ext = strrchr(filename, '.');
    int result = 0;
    
    if (ext != NULL) {
        if (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0) {
            result = stbi_write_png(filename, img.width, img.height, img.channels, data, img.width * img.channels);
        } else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 || 
                   strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0) {
            result = stbi_write_jpg(filename, img.width, img.height, img.channels, data, 90);
        } else if (strcmp(ext, ".bmp") == 0 || strcmp(ext, ".BMP") == 0) {
            result = stbi_write_bmp(filename, img.width, img.height, img.channels, data);
        } else {
            // Default to PNG
            result = stbi_write_png(filename, img.width, img.height, img.channels, data, img.width * img.channels);
        }
    } else {
        result = stbi_write_png(filename, img.width, img.height, img.channels, data, img.width * img.channels);
    }
    
    free(data);
    
    if (!result) {
        fprintf(stderr, "Failed to save image: %s\n", filename);
    } else {
        printf("Saved image: %s\n", filename);
    }
}

Image generateTestImage(unsigned int width, unsigned int height, unsigned int channels) {
    Image img = allocateImage(width, height, channels);
    
    // Generate a synthetic pattern (gradient + checkerboard)
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            size_t idx = (y * width + x) * channels;
            
            // Create interesting pattern
            float gradientX = (float)x / width;
            float gradientY = (float)y / height;
            int checkerSize = 32;
            float checker = ((x / checkerSize) % 2 == (y / checkerSize) % 2) ? 1.0f : 0.0f;
            
            for (unsigned int c = 0; c < channels; c++) {
                if (c == 0) { // Red/Gray
                    img.data[idx + c] = gradientX * 0.7f + checker * 0.3f;
                } else if (c == 1) { // Green
                    img.data[idx + c] = gradientY * 0.7f + checker * 0.3f;
                } else { // Blue
                    img.data[idx + c] = (gradientX + gradientY) * 0.35f + checker * 0.3f;
                }
            }
        }
    }
    
    return img;
}

Image* generateTestBatch(unsigned int count, unsigned int width, unsigned int height, unsigned int channels) {
    Image* batch = (Image*)malloc(count * sizeof(Image));
    
    if (batch == NULL) {
        FATAL("Unable to allocate batch memory");
    }
    
    for (unsigned int i = 0; i < count; i++) {
        batch[i] = generateTestImage(width, height, channels);
        
        // Add some variation to each image
        size_t numElements = width * height * channels;
        for (size_t j = 0; j < numElements; j++) {
            batch[i].data[j] += (float)(i % 10) * 0.01f;
            if (batch[i].data[j] > 1.0f) batch[i].data[j] = 1.0f;
        }
    }
    
    printf("Generated %u test images (%ux%u, %d channels)\n", count, width, height, channels);
    return batch;
}

void freeBatch(Image* batch, unsigned int count) {
    if (batch != NULL) {
        for (unsigned int i = 0; i < count; i++) {
            freeImage(batch[i]);
        }
        free(batch);
    }
}
