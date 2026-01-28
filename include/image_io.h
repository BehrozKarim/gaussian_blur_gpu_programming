#ifndef __IMAGE_IO_H__
#define __IMAGE_IO_H__

#include "support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load image from file (supports PNG, JPG, BMP, PPM)
Image loadImage(const char* filename);

// Save image to file
void saveImage(const char* filename, Image img);

// Generate synthetic test image
Image generateTestImage(unsigned int width, unsigned int height, unsigned int channels);

// Generate batch of test images
Image* generateTestBatch(unsigned int count, unsigned int width, unsigned int height, unsigned int channels);

// Free batch of images
void freeBatch(Image* batch, unsigned int count);

#ifdef __cplusplus
}
#endif

#endif // __IMAGE_IO_H__
