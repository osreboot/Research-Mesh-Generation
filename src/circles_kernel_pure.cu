#include "math.cuh"

__global__ void circlesKernelPure(bool *out, const double* __restrict__ px, const double* __restrict__ py,
                                  const Point p1, const Point p2, const Point p3, int pointsSize, const int step){
    unsigned int i = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x + threadIdx.x;
    for(; i < pointsSize; i += step){
        out[i] = det(p1.x - px[i], p1.y - py[i], (p1.x * p1.x - px[i] * px[i]) + (p1.y * p1.y - py[i] * py[i]),
                     p2.x - px[i], p2.y - py[i], (p2.x * p2.x - px[i] * px[i]) + (p2.y * p2.y - py[i] * py[i]),
                     p3.x - px[i], p3.y - py[i], (p3.x * p3.x - px[i] * px[i]) + (p3.y * p3.y - py[i] * py[i])) <= RADIUS_ERROR;
    }
}

class CirclesKernelPure : public Circles{

private:
    const double *px = nullptr, *py = nullptr;
    int pointsSize = 0;
    double *dpx = nullptr, *dpy = nullptr;
    bool *doutput = nullptr;

public:
    __host__ void load(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;

        CUDA_CHECK(cudaMalloc((void**)&dpx, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMalloc((void**)&dpy, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMemcpy(dpx, px, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpy, py, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) override {
        dim3 dimGrid = KERNEL_DIM_GRID;
        dim3 dimBlock = KERNEL_DIM_BLOCK;
        const int step = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x;

        circlesKernelPure<<<dimGrid, dimBlock>>>(doutput, dpx, dpy, p1, p2, p3, pointsSize, step);
        CUDA_CHECK_LAST_ERROR();
    }

    __host__ void save(bool *output) override {
        CUDA_CHECK(cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost));
    }

    __host__ void cleanup() override {
        cudaFree(dpx);
        cudaFree(dpy);
        cudaFree(doutput);
    }

    string getFileName() const override {
        return "kernel_pure";
    }

};
