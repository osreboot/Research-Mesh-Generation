#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "math.cuh"

// Simplified distance double-GEMM Tensor Core algorithm (WITHOUT post-processing kernel invocation for metrics gathering purposes)
class CirclesPTensorTwostep : public Circles{

private:
    int n = 0, m = 0, k = 0;

    float *px = nullptr, *py = nullptr, *ident = nullptr;
    int pointsSize = 0;
    float *pxy2 = nullptr;

    cublasHandle_t handle;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;

public:
    __host__ void load(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        pointsSize = pointsSizeArg;

        m = TENSOR_GET_M;
        n = TENSOR_GET_N;
        k = TENSOR_GET_K;

        px = new float[m * k];
        py = new float[m * n];
        pxy2 = new float[m * k];

        for(int i = 0; i < m * k; i++){
            if(i < pointsSize){
                px[i] = (float)pxArg[i];
                py[i] = (float)pyArg[i];
                pxy2[i] = (float)pxArg[i] * (float)pxArg[i] + (float)pyArg[i] * (float)pyArg[i];
            }else{
                px[i] = 0.0f;
                py[i] = 0.0f;
                pxy2[i] = 0.0f;
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_D), m * n * sizeof(d_D[0])));

        CUBLAS_CHECK(cublasSetVector(m * k, sizeof(px[0]), px, 1, d_A, 1));
        CUBLAS_CHECK(cublasSetVector(k * n, sizeof(ident[0]), ident, 1, d_B, 1));
        CUBLAS_CHECK(cublasSetVector(n * m, sizeof(py[0]), py, 1, d_C, 1));
        CUBLAS_CHECK(cublasSetVector(n * m, sizeof(pxy2[0]), pxy2, 1, d_D, 1));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) override {
        const Circumcircle circle(p1, p2, p3);
        const float coefPx = -2.0f * (float)circle.x;
        const float coefPy = -2.0f * (float)circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + RADIUS_ERROR;

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &coefPx,
                                  d_A, CUDA_R_32F, m,
                                  d_B, CUDA_R_32F, k, &coefPy,
                                  d_C, CUDA_R_32F, m,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const float alpha = 1.0f;
        const float beta = 1.0f;

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                                  d_C, CUDA_R_32F, m,
                                  d_B, CUDA_R_32F, k, &beta,
                                  d_D, CUDA_R_32F, m,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    __host__ void save(bool *output) override {}

    __host__ void cleanup() override {
        CUBLAS_CHECK(cublasDestroy(handle));

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);

        delete[] px;
        delete[] py;
        delete[] pxy2;
        delete[] ident;
    }

    string getFileName() const override {
        return "ptensor_twostep";
    }

};
