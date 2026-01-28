#!/bin/bash
#SBATCH --job-name=gaussian_blur
#SBATCH --account=project_2016196
#SBATCH --partition=gpusmall
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output=blur_benchmark_%j.out
#SBATCH --error=blur_benchmark_%j.err

# Load modules
module purge
module load gcc/11.2.0 cuda/11.5.0

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print job information
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="
echo ""

# Compile the code
echo "Compiling the code..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Create output directory
mkdir -p output_images

# ============================================================================
# Test Suite 1: Different Image Sizes (Fixed kernel, fixed batch)
# ============================================================================
echo "========================================="
echo "TEST SUITE 1: Image Size Scaling"
echo "========================================="
echo ""

echo "--- Test 1.1: Small (512x512) ---"
./blur_benchmark 512 512 7 2.0 10
echo ""

echo "--- Test 1.2: Medium (1024x1024) ---"
./blur_benchmark 1024 1024 7 2.0 10
echo ""

echo "--- Test 1.3: Large (2048x2048) ---"
./blur_benchmark 2048 2048 7 2.0 10
echo ""

echo "--- Test 1.4: 4K (3840x2160) ---"
./blur_benchmark 3840 2160 7 2.0 10
echo ""

# ============================================================================
# Test Suite 2: Different Kernel Sizes (Fixed image, fixed batch)
# ============================================================================
echo "========================================="
echo "TEST SUITE 2: Kernel Size Impact"
echo "========================================="
echo ""

echo "--- Test 2.1: 3x3 Kernel ---"
./blur_benchmark 1024 1024 3 1.0 10
echo ""

echo "--- Test 2.2: 5x5 Kernel ---"
./blur_benchmark 1024 1024 5 1.5 10
echo ""

echo "--- Test 2.3: 7x7 Kernel ---"
./blur_benchmark 1024 1024 7 2.0 10
echo ""

echo "--- Test 2.4: 9x9 Kernel ---"
./blur_benchmark 1024 1024 9 2.5 10
echo ""

echo "--- Test 2.5: 11x11 Kernel ---"
./blur_benchmark 1024 1024 11 3.0 10
echo ""

# ============================================================================
# Test Suite 3: Batch Size Scaling (Fixed image, fixed kernel)
# ============================================================================
echo "========================================="
echo "TEST SUITE 3: Batch Size Scaling"
echo "========================================="
echo ""

echo "--- Test 3.1: Batch Size 5 ---"
./blur_benchmark 1024 1024 7 2.0 5
echo ""

echo "--- Test 3.2: Batch Size 10 ---"
./blur_benchmark 1024 1024 7 2.0 10
echo ""

echo "--- Test 3.3: Batch Size 25 ---"
./blur_benchmark 1024 1024 7 2.0 25
echo ""

echo "--- Test 3.4: Batch Size 50 ---"
./blur_benchmark 1024 1024 7 2.0 50
echo ""

# ============================================================================
# Test Suite 4: Stress Test
# ============================================================================
echo "========================================="
echo "TEST SUITE 4: Stress Test"
echo "========================================="
echo ""

echo "--- Test 4.1: Large Image + Large Kernel + Large Batch ---"
./blur_benchmark 2048 2048 11 3.0 50
echo ""

echo "--- Test 4.2: 4K Image + Medium Kernel + Large Batch ---"
./blur_benchmark 3840 2160 9 2.5 25
echo ""

# Job completion
echo ""
echo "========================================="
echo "All tests completed successfully!"
echo "Job finished at: $(date)"
echo "========================================="

# Print GPU utilization summary
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
