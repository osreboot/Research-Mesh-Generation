#include "math.cuh"

__global__ void circlesPure(bool *out, const double* __restrict__ px, const double* __restrict__ py, const Point p1, const Point p2, const Point p3, int pointsSize, int threads){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < pointsSize; i += threads){
        out[i] = det(p1.x - px[i], p1.y - py[i], (p1.x * p1.x - px[i] * px[i]) + (p1.y * p1.y - py[i] * py[i]),
                     p2.x - px[i], p2.y - py[i], (p2.x * p2.x - px[i] * px[i]) + (p2.y * p2.y - py[i] * py[i]),
                     p3.x - px[i], p3.y - py[i], (p3.x * p3.x - px[i] * px[i]) + (p3.y * p3.y - py[i] * py[i])) <= 0.00000001;
    }
}

class CirclesKernelPure : public Circles{

private:
    const double *px = nullptr, *py = nullptr;
    double *dpx = nullptr, *dpy = nullptr;
    int pointsSize = 0;

public:
    __host__ void initialize(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;

        cudaMalloc((void**)&dpx, sizeof(double) * pointsSize);
        cudaMalloc((void**)&dpy, sizeof(double) * pointsSize);
        cudaMemcpy(dpx, px, sizeof(double) * pointsSize, cudaMemcpyHostToDevice);
        cudaMemcpy(dpy, py, sizeof(double) * pointsSize, cudaMemcpyHostToDevice);
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        bool *doutput;

        cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize);

        const int threads = 128;
        const int blocks = 128 * 128;

        circlesPure<<<blocks,threads>>>(doutput, dpx, dpy, p1, p2, p3, pointsSize, threads * blocks);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) cout << "ERROR (CUDA): " << cudaGetErrorString(err) << endl;

        cudaDeviceSynchronize();

        cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost);

        cudaFree(doutput);
    }

    __host__ void cleanup() override {
        cudaFree(dpx);
        cudaFree(dpy);
    }

    string getFileName() const override {
        return "kernel_pure";
    }

};
