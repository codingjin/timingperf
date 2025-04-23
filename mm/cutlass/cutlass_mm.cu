#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/tensor_view_io.h>

constexpr int ALIGN = 8;

int align(int n) {
    return (n + ALIGN - 1) / ALIGN * ALIGN;
}

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::cerr << "CUDA error " << err_ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// CUTLASS GEMM configuration
using ElementInputA = cutlass::half_t;              // Input A data type (half-precision)
using ElementInputB = cutlass::half_t;              // Input B data type (half-precision)
using ElementOutput = float;                       // Output data type (single-precision)
using ElementAccumulator = float;                  // Accumulator data type (single-precision)
using LayoutInputA = cutlass::layout::ColumnMajor;  // Layout for A
using LayoutInputB = cutlass::layout::ColumnMajor;  // Layout for B
using LayoutOutput = cutlass::layout::ColumnMajor;  // Layout for C/D

// Tensor Core (TensorOp) configuration
using MMAOp = cutlass::arch::OpClassTensorOp;       // Use Tensor Cores
using SmArch = cutlass::arch::Sm80;                // Target architecture: SM80 (Ampere)

// Threadblock tile size
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  // Threadblock tile size
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;           // Warp tile size
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;               // MMA operation tile size

// Epilogue operation
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Number of elements per vectorized memory access
    ElementAccumulator,                                // Accumulator data type
    ElementAccumulator>;                               // Data type for alpha/beta

// Threadblock swizzle
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipeline stages
constexpr int NumStages = 2;

// Instantiate the CUTLASS GEMM kernel
using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages>;

float get_median(float array[], int size) {
    std::vector<float> temp(array, array + size);
    std::sort(temp.begin(), temp.end());
    if (size % 2 == 1) {
        return temp[size / 2];
    } else {
        return (temp[size / 2 - 1] + temp[size / 2]) / 2.0f;
    }
}

void fill_random(std::vector<cutlass::half_t>& data, int size, float range = 1.0f) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<cutlass::half_t>(static_cast<float>(rand()) / RAND_MAX * range);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./cutlass_mm <M> <N> <K>" << std::endl;
        return EXIT_FAILURE;
    }

    const int M = align(atoi(argv[1]));
    const int N = align(atoi(argv[2]));
    const int K = align(atoi(argv[3]));

    std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Problem size
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Host memory allocation
    std::vector<cutlass::half_t> host_A(M * K);
    std::vector<cutlass::half_t> host_B(K * N);
    std::vector<float> host_C(M * N);

    // Fill host memory with random data
    fill_random(host_A, host_A.size());
    fill_random(host_B, host_B.size());
    std::fill(host_C.begin(), host_C.end(), 0.0f);

    // Device memory allocation
    cutlass::half_t* A;
    cutlass::half_t* B;
    float* C;
    CUDA_CHECK(cudaMalloc(&A, host_A.size() * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&B, host_B.size() * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&C, host_C.size() * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A, host_A.data(), host_A.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, host_B.data(), host_B.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C, host_C.data(), host_C.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Instantiate CUTLASS GEMM
    Gemm gemm_op;

    // Create arguments for GEMM
    typename Gemm::Arguments arguments{
        problem_size,
        {A, M},
        {B, K},
        {C, M},
        {C, M},
        {alpha, beta}
    };

    // Check if the problem size is supported
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM configuration not supported" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate workspace
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    // Warm-up
    for (int i = 0; i < 3; ++i) {
        status = gemm_op(arguments, workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed during warm-up" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Timing
    float time_array[9];
    cudaEvent_t start, stop;
    for (int i = 0; i < 9; ++i) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        status = gemm_op(arguments, workspace);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_array[i], start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed during timing" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Calculate median time
    float median_time = get_median(time_array, 9);
    std::cout << "Median time: " << median_time << " ms" << std::endl;

    // Calculate performance
    float gflops = 2.0f * M * N * K / (median_time * 1.0e6f);
    std::cout << "Performance: " << gflops << " GFLOPs" << std::endl;

    // Free memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }

    return 0;
}