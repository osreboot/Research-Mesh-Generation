#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "math.cuh"

__global__ void circlesKernelLightweight(bool *out, const double* __restrict__ dpx, const double* __restrict__ dpy, const double* __restrict__ dpxy2,
                               const double coefPx, const double coefPy, const double comp, const int pointsSize, const int step){
    unsigned int i = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x + threadIdx.x;
    for(; i < pointsSize; i += step){
        out[i] = coefPx * dpx[i] + coefPy * dpy[i] + dpxy2[i] <= comp;
    }
}

// Simplified distance-based parallel algorithm
class CirclesKernelLightweight : public Circles{

private:
    const double *px = nullptr, *py = nullptr;
    int pointsSize = 0;
    double *dpx = nullptr, *dpy = nullptr, *pxy2 = nullptr, *dpxy2 = nullptr;
    bool *doutput = nullptr;

public:
    __host__ void load(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;

        pxy2 = new double[pointsSize];

        for(int i = 0; i < pointsSize; i++){
            pxy2[i] = px[i] * px[i] + py[i] * py[i];
        }

        CUDA_CHECK(cudaMalloc((void**)&dpx, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMalloc((void**)&dpy, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMalloc((void**)&dpxy2, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMemcpy(dpx, px, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpy, py, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpxy2, pxy2, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) override {
        const Circumcircle circle(p1, p2, p3);
        const double coefPx = -2.0 * circle.x;
        const double coefPy = -2.0 * circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + RADIUS_ERROR;

        dim3 dimGrid = KERNEL_DIM_GRID;
        dim3 dimBlock = KERNEL_DIM_BLOCK;
        const int step = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x;

        circlesKernelLightweight<<<dimGrid, dimBlock>>>(doutput, dpx, dpy, dpxy2, coefPx, coefPy, comp, pointsSize, step);
        CUDA_CHECK_LAST_ERROR();
    }

    __host__ void save(bool *output) override {
        CUDA_CHECK(cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost));
    }

    __host__ void cleanup() override {
        cudaFree(dpx);
        cudaFree(dpy);
        cudaFree(dpxy2);
        cudaFree(doutput);

        delete[] pxy2;
    }

    string getFileName() const override {
        return "kernel_lightweight";
    }

};
