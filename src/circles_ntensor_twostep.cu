#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "math.cuh"

__global__ void circlesNTensorTwostep(bool *out, const float* __restrict__ d_D, const double comp, const int pointsSize, const int step){
    unsigned int i = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x + threadIdx.x;
    for(; i < pointsSize; i += step){
        out[i] = static_cast<double>(d_D[i]) <= comp;
    }
}

// Simplified distance double-GEMM Tensor Core algorithm (WITHOUT Tensor Core invocation for metrics gathering purposes)
class CirclesNTensorTwostep : public Circles{

private:
    int n = 0, m = 0, k = 0;

    float *px = nullptr, *py = nullptr, *ident = nullptr;
    int pointsSize = 0;
    bool *doutput = nullptr;

    float *d_D = nullptr;

public:
    __host__ void load(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        pointsSize = pointsSizeArg;

        m = TENSOR_GET_M;
        n = TENSOR_GET_N;
        k = TENSOR_GET_K;

        px = new float[m * k];
        py = new float[m * n];

        for(int i = 0; i < m * k; i++){
            if(i < pointsSize){
                px[i] = (float)pxArg[i];
                py[i] = (float)pyArg[i];
            }else{
                px[i] = 0.0f;
                py[i] = 0.0f;
            }
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_D), m * n * sizeof(d_D[0])));
        CUBLAS_CHECK(cublasSetVector(n * m, sizeof(py[0]), py, 1, d_D, 1));

        CUDA_CHECK(cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) override {
        const Circumcircle circle(p1, p2, p3);
        const float coefPx = -2.0f * (float)circle.x;
        const float coefPy = -2.0f * (float)circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + RADIUS_ERROR;

        // Tensor core instructions would be invoked here, to theoretically save compute time for the kernel

        dim3 dimGrid = KERNEL_DIM_GRID;
        dim3 dimBlock = KERNEL_DIM_BLOCK;
        const int step = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x;

        circlesNTensorTwostep<<<dimGrid, dimBlock>>>(doutput, d_D, comp, pointsSize, step);
        CUDA_CHECK_LAST_ERROR();
    }

    __host__ void save(bool *output) override {
        CUDA_CHECK(cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost));
    }

    __host__ void cleanup() override {
        cudaFree(d_D);
        cudaFree(doutput);

        delete[] px;
        delete[] py;
        delete[] ident;
    }

    string getFileName() const override {
        return "ntensor_twostep";
    }

};
