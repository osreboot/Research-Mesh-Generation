#pragma once

#include "math.cuh"

__global__ void circlesNoop(bool *out, const Point* __restrict__ points, const Point p1, const Point p2, const Point p3, int pointsSize, int threads){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < pointsSize; i += threads){
        out[i] = false;
    }
}

class CirclesKernelNoop : public Circles{

private:
    const Point *points = nullptr;
    Point *dpoints = nullptr;
    int pointsSize = 0;

public:
    __host__ void initialize(const Point *pointsArg, int pointsSizeArg) override {
        points = pointsArg;
        pointsSize = pointsSizeArg;

        cudaMalloc((void**)&dpoints, sizeof(Point) * pointsSize);
        cudaMemcpy(dpoints, points, sizeof(Point) * pointsSize, cudaMemcpyHostToDevice);
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        bool *doutput;

        cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize);

        const int threads = 128;
        const int blocks = 128 * 128;

        circlesNoop<<<blocks,threads>>>(doutput, dpoints, p1, p2, p3, pointsSize, threads * blocks);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) cout << "ERROR (CUDA): " << cudaGetErrorString(err) << endl;

        cudaDeviceSynchronize();

        cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost);

        cudaFree(doutput);
    }

    __host__ void cleanup() override {
        cudaFree(dpoints);
    }

    string getFileName() const override {
        return "kernel_noop";
    }

};
