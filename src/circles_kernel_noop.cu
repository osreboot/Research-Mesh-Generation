#include "math.cuh"

__global__ void circlesNoop(bool *out, const double* __restrict__ px, const double* __restrict__ py, const Point p1, const Point p2, const Point p3, int pointsSize, int threads){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < pointsSize; i += threads){
        out[i] = false;
    }
}

class CirclesKernelNoop : public Circles{

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

        circlesNoop<<<blocks,threads>>>(doutput, dpx, dpy, p1, p2, p3, pointsSize, threads * blocks);
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
        return "kernel_noop";
    }

};
