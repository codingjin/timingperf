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
#include <cuda_fp16.h>

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

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}


void cpu_fill_rand(__half *A, int m, int n)
{
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j) {
            A[i*n+j] = __half(((float)rand()/(float)(RAND_MAX)));
            //printf("A=%f\n", __half2float(A[i*n+j]));
        }
}

void cpu_fill_zero(__half *A, int m, int n)
{
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            A[i*n+j] = (__half)0;
}

void verify_solution(__half *a, __half *b, __half *c, int M, int N, int K) {
    float temp;
    float epsilon = 10;

    for (int i=0; i<M; ++i) {
        for (int j=0; j<N; ++j) {
            temp = 0;
            for (int k=0; k<K; ++k) {
                temp += __half2float(a[i*K + k]) * __half2float(b[k*N+j]);
            }
            std::cout << "i=" << i << " j=" << j << " c=" << __half2float(c[i*N+j]) << " temp=" << temp << std::endl;
            assert(fabs(__half2float(c[i*N+j])-temp) < epsilon);
        }
    }
}


int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cout << "Invalid input" << std::endl;
        std::cout << "Usage: ./cublas_mm <M> <N> <K>" << std::endl;
        return EXIT_FAILURE;
    }

    __half *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int M, N, K;

    cudaEvent_t start, stop;
    float elapsedTime;
    float time_array[9];

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    __half alpha = 1.0;
    __half beta = 0.0;

    if (M<=0 || N<=0 || K<=0) {
        std::cout << "Invalid M N K" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;

    cublasStatus_t stat;
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    srand(149);
    h_A = (__half*)malloc(sizeof(__half)*M*K);
    cpu_fill_rand(h_A, M, K);
    h_B = (__half*)malloc(sizeof(__half)*K*N);
    cpu_fill_rand(h_B, K, N);
    h_C = (__half*)malloc(sizeof(__half)*M*N);
    cpu_fill_zero(h_C, M, N);

    checkCuda(cudaMalloc((void**)&d_A, sizeof(__half)*M*K));
    checkCuda(cudaMalloc((void**)&d_B, sizeof(__half)*K*N));
    checkCuda(cudaMalloc((void**)&d_C, sizeof(__half)*M*N));
    
    checkCuda(cudaMemcpy(d_A, h_A, sizeof(__half)*M*K, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, sizeof(__half)*K*N, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C, h_C, sizeof(__half)*M*N, cudaMemcpyHostToDevice));

    // Warmup - 3 runs
    for (int i=0; i<3; ++i) {
        stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
            &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaDeviceSynchronize();
        if(stat != CUBLAS_STATUS_SUCCESS){
            std::cerr << "cublasSgemmBatched failed" << std::endl;
            exit(1);
        }
        assert(!cudaGetLastError());
    }

    // 9 runs
    for (int i=0; i<9; ++i) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
            &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaDeviceSynchronize();
        time_array[i] = elapsedTime;
    }

    float median_time = get_median(time_array, 9);
    std::cout << "Median_time = " << median_time << std::endl;
    std::cout << "Performance is " << 2.0e-6*M*N*K / median_time << " GFLOPs" << std::endl << std::endl;

    return 0;
}


