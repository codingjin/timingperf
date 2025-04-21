#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


using data_type = __half;

// Comparison function for qsort
int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

float get_median(float array[], int size) {
    // Make a copy to avoid modifying original array
    float* temp = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        temp[i] = array[i];
        //printf("%d-th : %f\n", i, (double)temp[i]);
    }
    
    // Sort the array
    qsort(temp, size, sizeof(float), compare_floats);
    
    // Get median
    float median;
    if (size % 2 == 1) {
        // Odd number of elements - return middle element
        median = temp[size/2];
    } else {
        // Even number - return average of two middle elements
        median = (temp[size/2 - 1] + temp[size/2]) / 2.0f;
    }
    
    free(temp);
    return median;
}

void cpu_fill_rand(std::vector<std::vector<data_type>>& array, int m, int n)
{
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            array[i][j] = data_type((float)rand()/(float)(RAND_MAX));
}

void cpu_fill_zero(std::vector<std::vector<data_type>>& array, int m, int n)
{
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            array[i][j] = data_type(0);
}
 

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cout << "Invalid input" << std::endl;
        std::cout << "Usage: ./cublas_mm <batch_size> <M> <N> <K>" << std::endl;
        return EXIT_FAILURE;
    }

    cublasHandle_t cublasH = NULL;
    //cudaStream_t stream = NULL;

    const int batch_size = atoi(argv[1]);
    const int M = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const int K = atoi(argv[4]);
    std::cout << "batch_size=" << batch_size << " M=" << M << " N=" << N << " K=" << K << std::endl;
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    std::vector<std::vector<data_type>> A_array(batch_size, std::vector<data_type>(M*K));
    std::vector<std::vector<data_type>> B_array(batch_size, std::vector<data_type>(K*N));
    std::vector<std::vector<data_type>> C_array(batch_size, std::vector<data_type>(M*N));
    cpu_fill_rand(A_array, batch_size, M*K);
    cpu_fill_rand(B_array, batch_size, K*N);
    cpu_fill_zero(C_array, batch_size, M*N);

    data_type **d_A_array = nullptr;
    data_type **d_B_array = nullptr;
    data_type **d_C_array = nullptr;

    std::vector<data_type *> d_A(batch_size, nullptr);
    std::vector<data_type *> d_B(batch_size, nullptr);
    std::vector<data_type *> d_C(batch_size, nullptr);

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(data_type) * A_array[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(data_type) * B_array[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * C_array[i].size()));
    }
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * batch_size));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * batch_size));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * batch_size));

    for (int i = 0; i < batch_size; i++) {
        /*
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], A_array[i].data(), sizeof(data_type) * A_array[i].size(),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], B_array[i].data(), sizeof(data_type) * B_array[i].size(),
                                   cudaMemcpyHostToDevice, stream));
        */
        CUDA_CHECK(cudaMemcpy(d_A[i], A_array[i].data(), sizeof(data_type) * A_array[i].size(),
                                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B[i], B_array[i].data(), sizeof(data_type) * B_array[i].size(),
                                   cudaMemcpyHostToDevice));
    }
    /*
    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type *) * batch_size,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * batch_size,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_size,
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    */
    CUDA_CHECK(cudaMemcpy(d_A_array, d_A.data(), sizeof(data_type *) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_size, cudaMemcpyHostToDevice));

    //CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    float elapsedTime;
    float time_array[9];
    /* step 3: compute */
    // warmup -3 rounds
    for (int i=0; i<3; ++i) {
        CUBLAS_CHECK(cublasHgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_array, N,
                                d_A_array, K, &beta, d_C_array, N, batch_size));
        cudaDeviceSynchronize();
        assert(!cudaGetLastError());
    }

    // measure
    for (int i=0; i<9; ++i) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        CUBLAS_CHECK(cublasHgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_array, N,
            d_A_array, K, &beta, d_C_array, N, batch_size));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaDeviceSynchronize();
        time_array[i] = elapsedTime;
    }
    
    float median_time = get_median(time_array, 9);
    std::cout << "Median_time = " << median_time << std::endl;
    std::cout << "Performance is " << 2.0e-6*batch_size*M*N*K / median_time << " GFLOPs" << std::endl << std::endl;

    /* step 4: copy data to host */
    /*
    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaMemcpyAsync(C_array[i].data(), d_C[i], sizeof(data_type) * C_array[i].size(),
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    */


    return 0;
}