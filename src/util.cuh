#pragma once

using namespace std;

// The number of unique triangles to evaluate for the circumcircle tests
#define CIRCLE_TEST_TRIANGLES 100

// Error constants used to fix rounding issues
#define RADIUS_ERROR (0.0000000001)
#define RADIUS_FLOAT_ERROR (0.0000001)

// Grid and block dimensions for all kernels
#define KERNEL_DIM_GRID {84, 64, 1}
#define KERNEL_DIM_BLOCK {64, 1, 1}

// GEMM matrix dimensions
#define TENSOR_GET_M (4096)
#define TENSOR_GET_N ((int)ceil(((double)pointsSize / TENSOR_GET_M) / 16.0) * 16)
#define TENSOR_GET_K ((int)ceil(((double)pointsSize / TENSOR_GET_M) / 16.0) * 16)

#define CUDA_CHECK(errArg)                                                              \
    do{                                                                                 \
        cudaError_t err = (errArg);                                                     \
        if(err != cudaSuccess){                                                         \
            printf("ERROR (CUDA): %d at %s:%d\n", err, __FILE__, __LINE__);             \
            throw runtime_error("ERROR (CUDA)");                                        \
        }                                                                               \
    }while(false)

#define CUBLAS_CHECK(errArg)                                                            \
    do{                                                                                 \
        cublasStatus_t err = (errArg);                                                  \
        if(err != CUBLAS_STATUS_SUCCESS){                                               \
            printf("ERROR (CUBLAS): %d at %s:%d\n", err, __FILE__, __LINE__);           \
            throw runtime_error("ERROR (CUBLAS)");                                      \
        }                                                                               \
    }while(false)

#define CUDA_CHECK_LAST_ERROR()                                                         \
    do{                                                                                 \
        cudaError_t err = cudaGetLastError();                                           \
        if(err != cudaSuccess){                                                         \
            printf("ERROR (CUDA): %d at %s:%d\n", err, __FILE__, __LINE__);             \
            throw runtime_error("ERROR (CUDA)");                                        \
        }                                                                               \
    }while(false)