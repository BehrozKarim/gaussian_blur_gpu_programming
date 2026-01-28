NVCC        = nvcc
CXX         = g++
NVCC_FLAGS  = -O3 -arch=sm_70 -Xcompiler -fopenmp
CXX_FLAGS   = -O3 -fopenmp
INCLUDES    = -I./include -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 -Xcompiler -fopenmp -lm
EXE         = blur_benchmark
OBJ_DIR     = obj

# Source files
CUDA_SRCS   = src/main.cu src/support.cu src/kernels.cu
CPP_SRCS    = src/image_io.cpp src/cpu_blur.cpp

# Object files
CUDA_OBJS   = $(OBJ_DIR)/main.o $(OBJ_DIR)/support.o $(OBJ_DIR)/kernels.o
CPP_OBJS    = $(OBJ_DIR)/image_io.o $(OBJ_DIR)/cpu_blur.o
ALL_OBJS    = $(CUDA_OBJS) $(CPP_OBJS)

# Create obj directory if it doesn't exist
$(shell mkdir -p $(OBJ_DIR))

default: $(EXE)

# Compile CUDA files
$(OBJ_DIR)/main.o: src/main.cu include/support.h include/image_io.h include/kernels.cuh
	$(NVCC) -c -o $@ src/main.cu $(NVCC_FLAGS) $(INCLUDES)

$(OBJ_DIR)/support.o: src/support.cu include/support.h
	$(NVCC) -c -o $@ src/support.cu $(NVCC_FLAGS) $(INCLUDES)

$(OBJ_DIR)/kernels.o: src/kernels.cu include/kernels.cuh include/support.h
	$(NVCC) -c -o $@ src/kernels.cu $(NVCC_FLAGS) $(INCLUDES)

# Compile C++ files
$(OBJ_DIR)/image_io.o: src/image_io.cpp include/image_io.h include/support.h
	$(CXX) -c -o $@ src/image_io.cpp $(CXX_FLAGS) $(INCLUDES)

$(OBJ_DIR)/cpu_blur.o: src/cpu_blur.cpp include/support.h
	$(CXX) -c -o $@ src/cpu_blur.cpp $(CXX_FLAGS) $(INCLUDES)

# Link
$(EXE): $(ALL_OBJS)
	$(NVCC) $(ALL_OBJS) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf $(OBJ_DIR)/*.o $(EXE) output_images/*.png

run: $(EXE)
	./$(EXE) 1024 1024 7 2.0 10

test_small: $(EXE)
	./$(EXE) 512 512 5 1.5 5

test_large: $(EXE)
	./$(EXE) 2048 2048 9 3.0 20

test_4k: $(EXE)
	./$(EXE) 3840 2160 7 2.0 10

.PHONY: default clean run test_small test_large test_4k
