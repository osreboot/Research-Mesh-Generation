#pragma once

#include <mma.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "math.cuh"

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

class CirclesTensorDisjoint : public Circles{

private:
    const Point *points = nullptr;
    int pointsSize = 0;

    cublasHandle_t handle;

public:
    __host__ void initialize(const Point *pointsArg, int pointsSizeArg) override {
        points = pointsArg;
        pointsSize = pointsSizeArg;

        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        /*

         need to compute:

         pxy2 = p.x * p.x + p.y * p.y

         a = a.x - p.x
         b = a.y - p.y
         c = a.x^2 + a.y^2 - pxy2
         d = b.x - p.x
         e = b.y - p.y
         f = b.x^2 + b.y^2 - pxy2
         g = c.y - p.y
         h = c.y - p.y
         i = c.x^2 + c.y^2 - pxy2



         */


        if(pointsSize < 512 * 512){
            for(int i = 0; i < pointsSize; i++){
                output[i] = false;
            }
            return;
        }

        const Circumcircle circle(p1, p2, p3);
        float coefPx = -2.0f * (float)circle.x;
        float coefPy = -2.0f * (float)circle.y;

        float *h_A, *h_B, *h_C;

        h_A = reinterpret_cast<float*>(malloc(512 * 512 * sizeof(h_A[0])));
        h_B = reinterpret_cast<float*>(malloc(512 * 512 * sizeof(h_B[0])));
        h_C = reinterpret_cast<float*>(malloc(512 * 512 * sizeof(h_C[0])));

        for(int i = 0; i < 512 * 512; i++){
            //h_A[i] = (float)points[i].x;
            //h_B[i] = (float)points[i].x;
            //h_C[i] = (float)points[i].x;
            h_C[i] = 0.0f;
            h_B[i] = 1.0f;
            h_A[i] = i < 512 ? (float)i : 0.0f;
        }

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), 512 * 512 * sizeof(d_A[0])));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), 512 * 512 * sizeof(d_B[0])));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), 512 * 512 * sizeof(d_C[0])));

        CUBLAS_CHECK(cublasSetVector(512 * 512, sizeof(h_A[0]), h_A, 1, d_A, 1));
        CUBLAS_CHECK(cublasSetVector(512 * 512, sizeof(h_B[0]), h_B, 1, d_B, 1));
        CUBLAS_CHECK(cublasSetVector(512 * 512, sizeof(h_C[0]), h_C, 1, d_C, 1));

        float alpha = 1.0f;
        float beta = 1.0f;

        cout << "==========" << endl;
        for(int i = 0; i < 3; i++) cout << alpha << " * " << h_A[i] << " * " << h_B[i] << " + " << beta << " * " << h_C[i] << endl;

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 512, 512, 512, &alpha,
                                  d_A, CUDA_R_32F, 512,
                                  d_B, CUDA_R_32F, 512, &beta,
                                  d_C, CUDA_R_32F, 512,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        h_C = reinterpret_cast<float*>(malloc(512 * 512 * sizeof(h_C[0])));

        CUBLAS_CHECK(cublasGetVector(512 * 512, sizeof(h_C[0]), d_C, 1, h_C, 1));

        cudaDeviceSynchronize();

        for(int i = 0; i < 10; i++){
            for(int j = 0; j < 10; j++) cout << h_C[j * 512 + i] << " ";
            cout << endl;
        }

        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    __host__ void cleanup() override {
        CUBLAS_CHECK(cublasDestroy(handle));
    }

    string getFileName() const override {
        return "tensor_disjoint";
    }

};
