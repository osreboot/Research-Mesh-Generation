#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "math.cuh"

__global__ void circlesTensorTwostepEC(bool *out, const double* __restrict__ dpx, const double* __restrict__ dpy, const double* __restrict__ dpxy2,
                                       const double doubleCoefPx, const double doubleCoefPy,
                                       const float* __restrict__ d_D, const double comp, const int pointsSize, const int step){
    unsigned int i = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x + threadIdx.x;
    for(; i < pointsSize; i += step){
        const double diff = comp - (double)d_D[i];
        if(fabs(diff) < 0.000001){
            out[i] = doubleCoefPx * dpx[i] + doubleCoefPy * dpy[i] + dpxy2[i] <= comp;
        }else out[i] = diff >= 0.0;
    }
}

// Simplified distance single-GEMM Tensor Core algorithm with precision error correction
class CirclesTensorTwostepEC : public Circles{

private:
    int n = 0, m = 0, k = 0;

    float *px = nullptr, *py = nullptr, *ident = nullptr;
    int pointsSize = 0;
    float *pxy2 = nullptr;
    const double *doubleDPx = nullptr, *doubleDPy = nullptr;
    double *dpx, *dpy, *doublePxy2 = nullptr, *dpxy2 = nullptr;
    bool *doutput = nullptr;

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

        doubleDPx = pxArg;
        doubleDPy = pyArg;
        doublePxy2 = new double[pointsSize];

        for(int i = 0; i < m * k; i++){
            if(i < pointsSize){
                px[i] = (float)pxArg[i];
                py[i] = (float)pyArg[i];
                pxy2[i] = (float)pxArg[i] * (float)pxArg[i] + (float)pyArg[i] * (float)pyArg[i];
                doublePxy2[i] = pxArg[i] * pxArg[i] + pyArg[i] * pyArg[i];
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

        CUDA_CHECK(cudaMalloc((void**)&dpx, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMalloc((void**)&dpy, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMemcpy(dpx, doubleDPx, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpy, doubleDPy, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&dpxy2, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMemcpy(dpxy2, doublePxy2, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) override {
        const Circumcircle circle(p1, p2, p3);
        const float coefPx = -2.0f * (float)circle.x;
        const float coefPy = -2.0f * (float)circle.y;
        const double doubleCoefPx = -2.0 * circle.x;
        const double doubleCoefPy = -2.0 * circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + RADIUS_ERROR;

        CUBLAS_CHECK(cublasSetVector(n * m, sizeof(py[0]), py, 1, d_C, 1));
        CUBLAS_CHECK(cublasSetVector(n * m, sizeof(pxy2[0]), pxy2, 1, d_D, 1));

        cudaDeviceSynchronize();

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &coefPx,
                                  d_A, CUDA_R_32F, m,
                                  d_B, CUDA_R_32F, k, &coefPy,
                                  d_C, CUDA_R_32F, m,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const float alpha = 1.0f;
        const float beta = 1.0f;

        cudaDeviceSynchronize();

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                                  d_C, CUDA_R_32F, m,
                                  d_B, CUDA_R_32F, k, &beta,
                                  d_D, CUDA_R_32F, m,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        dim3 dimGrid = KERNEL_DIM_GRID;
        dim3 dimBlock = KERNEL_DIM_BLOCK;
        const int step = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x;

        cudaDeviceSynchronize();

        circlesTensorTwostepEC<<<dimGrid, dimBlock>>>(doutput, dpx, dpy, dpxy2, doubleCoefPx, doubleCoefPy, d_D, comp, pointsSize, step);
        CUDA_CHECK_LAST_ERROR();
    }

    __host__ void save(bool *output) override {
        CUDA_CHECK(cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost));
    }

    __host__ void cleanup() override {
        CUBLAS_CHECK(cublasDestroy(handle));

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);

        cudaFree(dpx);
        cudaFree(dpy);
        cudaFree(dpxy2);

        cudaFree(doutput);

        delete[] px;
        delete[] py;
        delete[] pxy2;
        delete[] doublePxy2;
        delete[] ident;
    }

    string getFileName() const override {
        return "tensor_twostep";
    }

};
