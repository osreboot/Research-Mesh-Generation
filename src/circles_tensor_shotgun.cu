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

__global__ void circlesAssignOutput(bool *out, const float* d_D, const double* __restrict__ dpxy2, const double comp, int pointsSize, int threads){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < pointsSize; i += threads * blockDim.x){
        out[i] = (double)d_D[i] + dpxy2[i] <= comp;
    }
}

class CirclesTensorShotgun : public Circles{

private:
    int pointsSize = 0, n = 0, m = 0, k = 0;

    float *px = nullptr, *py = nullptr, *ident = nullptr;
    double *pxy2 = nullptr, *dpxy2 = nullptr;
    bool *doutput;

    cublasHandle_t handle;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

public:
    __host__ void initialize(const Point *pointsArg, int pointsSizeArg) override {
        pointsSize = pointsSizeArg;

        m = 256;
        n = 256;
        k = 256;
        while(m * k < pointsSize) m *= 2;
        //cout << "Initialized N to: " << n << endl;

        // n -> minimum 64, ideally 256

        px = new float[m * k];
        py = new float[m * n];
        pxy2 = new double[pointsSize];

        for(int i = 0; i < m * k; i++){
            if(i < pointsSize){
                px[i] = (float)pointsArg[i].x;
                py[i] = (float)pointsArg[i].y;
                pxy2[i] = pointsArg[i].x * pointsArg[i].x + pointsArg[i].y * pointsArg[i].y;
            }else{
                px[i] = 0.0f;
                py[i] = 0.0f;
                pxy2[i] = 0.0;
            }
        }

        ident = new float[k * n];
        for(int x = 0; x < n; x++){
            for(int y = 0; y < k; y++){
                ident[y * n + x] = (x == y) ? 1.0f : 0.0f;
            }
        }

        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), m * k * sizeof(d_A[0])));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), k * n * sizeof(d_B[0])));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), m * n * sizeof(d_C[0])));

        CUBLAS_CHECK(cublasSetVector(m * k, sizeof(px[0]), px, 1, d_A, 1));
        CUBLAS_CHECK(cublasSetVector(k * n, sizeof(ident[0]), ident, 1, d_B, 1));
        CUBLAS_CHECK(cublasSetVector(n * m, sizeof(py[0]), py, 1, d_C, 1));

        CUDA_CHECK(cudaMalloc((void**)&dpxy2, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMemcpy(dpxy2, pxy2, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        const Circumcircle circle(p1, p2, p3);
        const float coefPx = -2.0f * (float)circle.x;
        const float coefPy = -2.0f * (float)circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + 0.0000001;

        //cout << "cpx:  " << coefPx << "  cpy:  " << coefPy << endl;

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &coefPx,
                                  d_A, CUDA_R_32F, k,
                                  d_B, CUDA_R_32F, n, &coefPy,
                                  d_C, CUDA_R_32F, n,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));


        //CUBLAS_CHECK(cublasGetVector(n * n, sizeof(h_D[0]), d_C, 1, h_D, 1));

        /*
        for(int i = 0; i < 10; i++){
            cout << "x:   " << px[i] << "   y:   " << py[i] << endl;
        }

        for(int i = 0; i < 10; i++){
            for(int j = 0; j < 10; j++) cout << h_D[j * n + i] << " ";
            cout << endl;
        }*/

        //for(int i = 0; i < pointsSize; i++){
        //    output[i] = (double)h_D[i] + pxy2[i] <= comp;
        //}

        const int dim = min(max((int)ceil((sqrt((float)pointsSize)) / 32.0f) * 32, 32), 768);

        circlesAssignOutput<<<dim, dim>>>(doutput, d_C, dpxy2, comp, pointsSize, dim);
        CUDA_CHECK(cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost));
    }

    __host__ void cleanup() override {
        CUBLAS_CHECK(cublasDestroy(handle));

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cudaFree(dpxy2);
        cudaFree(doutput);

        delete[] px;
        delete[] py;
        delete[] pxy2;
        delete[] ident;
    }

    string getFileName() const override {
        return "tensor_shotgun";
    }

};
