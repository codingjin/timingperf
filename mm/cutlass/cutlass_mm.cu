#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// Comparison function for qsort
int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

float get_median(float array[], int size)
{
    // Make a copy to avoid modifying original array
    float *temp = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        temp[i] = array[i];
        // printf("%d-th : %f\n", i, (double)temp[i]);
    }

    // Sort the array
    qsort(temp, size, sizeof(float), compare_floats);

    // Get median
    float median;
    if (size % 2 == 1)
    {
        // Odd number of elements - return middle element
        median = temp[size / 2];
    }
    else
    {
        // Even number - return average of two middle elements
        median = (temp[size / 2 - 1] + temp[size / 2]) / 2.0f;
    }

    free(temp);
    return median;
}

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                  // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = float;                       // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<256, 128, 32>; // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                      // memory access. For a byte, it's 16
                                                      // elements. This becomes the vector width of
                                                      // math instructions in the epilogue too
    ElementAccumulator,                               // <- data type of accumulator
    ElementComputeEpilogue>;                          // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

int run()
{

    const int length_m = 5120;
    const int length_n = 4096;
    const int length_k = 4096;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk()); // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn()); // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn()); // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // reference kernel

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0); // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0); // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0); // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view()); // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,          // <- problem size of matrix multiplication
                                       tensor_a.device_ref(), // <- reference to matrix A on device
                                       tensor_b.device_ref(), // <- reference to matrix B on device
                                       tensor_c.device_ref(), // <- reference to matrix C on device
                                       tensor_d.device_ref(), // <- reference to matrix D on device
                                       {alpha, beta},         // <- tuple of alpha and beta
                                       split_k_slices};       // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device;

    // Launch device reference gemm kernel
    gemm_device(problem_size,
                alpha,
                tensor_a.device_ref(),
                tensor_b.device_ref(),
                beta,
                tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0 : -1);
}

float timing(int M, int N, int K)
{
    const int length_m = M;
    const int length_n = N;
    const int length_k = K;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk()); // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn()); // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn()); // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                            // reference kernel

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0); // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0); // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0); // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view()); // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,          // <- problem size of matrix multiplication
                                       tensor_a.device_ref(), // <- reference to matrix A on device
                                       tensor_b.device_ref(), // <- reference to matrix B on device
                                       tensor_c.device_ref(), // <- reference to matrix C on device
                                       tensor_d.device_ref(), // <- reference to matrix D on device
                                       {alpha, beta},         // <- tuple of alpha and beta
                                       split_k_slices};       // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device;

    // Launch device reference gemm kernel and record launch time
    cudaEvent_t start, stop;
    float elapsedTime;
    float time_array[9];
    // Warmup - 3 runs
    for (int i = 0; i < 3; ++i)
    {
        gemm_device(problem_size,
                    alpha,
                    tensor_a.device_ref(),
                    tensor_b.device_ref(),
                    beta,
                    tensor_c.device_ref(),
                    tensor_ref_d.device_ref());

        // Wait for kernels to finish
        cudaDeviceSynchronize();

        // Copy output data from CUTLASS and reference kernel to host for comparison
        tensor_d.sync_host();
        tensor_ref_d.sync_host();

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        bool passed = cutlass::reference::host::TensorEquals(
            tensor_d.host_view(),
            tensor_ref_d.host_view());

        std::cout << (passed ? "Passed" : "Failed") << std::endl;
    }

    // 9 runs
    for (int i = 0; i < 9; ++i)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemm_device(problem_size,
                    alpha,
                    tensor_a.device_ref(),
                    tensor_b.device_ref(),
                    beta,
                    tensor_c.device_ref(),
                    tensor_ref_d.device_ref());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaDeviceSynchronize();
        time_array[i] = elapsedTime;
    }

    float median_time = get_median(time_array, 9);
    std::cout << "Median_time = " << median_time << std::endl;
    std::cout << "Performance is " << 2.0e-6 * M * N * K / median_time << " GFLOPs" << std::endl
              << std::endl;

    return median_time;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cout << "Invalid input" << std::endl;
        std::cout << "Usage: ./cutlass_mm <M> <N> <K>" << std::endl;
        return EXIT_FAILURE;
    }
    int M, N, K;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    bool notSupported = false;

    // Turing Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)))
    {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (!((props.major * 10 + props.minor) >= 75))
    {
        std::cerr << "Turing Tensor Core operations must be run on a machine with compute capability at least 75."
                  << std::endl;

        notSupported = true;
    }

    if (notSupported)
    {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    timing(M, N, K);

    return 0;
}