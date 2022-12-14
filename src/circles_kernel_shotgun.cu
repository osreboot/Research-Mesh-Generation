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

__global__ void circlesShotgun(bool *out, const double* __restrict__ dpx, const double* __restrict__ dpy, const double* __restrict__ dpxy2,
                               const double coefPx, const double coefPy, const double comp, int pointsSize, int threads){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < pointsSize; i += threads * blockDim.x){
        out[i] = coefPx * dpx[i] + coefPy * dpy[i] + dpxy2[i] <= comp;
    }
}

class CirclesKernelShotgun : public Circles{

private:
    int pointsSize = 0;

    double *px = nullptr, *py = nullptr, *dpx = nullptr, *dpy = nullptr;
    double *pxy2 = nullptr, *dpxy2 = nullptr;
    bool *doutput;

public:
    __host__ void initialize(const Point *pointsArg, int pointsSizeArg) override {
        pointsSize = pointsSizeArg;

        px = new double[pointsSize];
        py = new double[pointsSize];
        pxy2 = new double[pointsSize];

        for(int i = 0; i < pointsSize; i++){
            px[i] = pointsArg[i].x;
            py[i] = pointsArg[i].y;
            pxy2[i] = pointsArg[i].x * pointsArg[i].x + pointsArg[i].y * pointsArg[i].y;
        }

        CUDA_CHECK(cudaMalloc((void**)&dpx, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMalloc((void**)&dpy, sizeof(double) * pointsSize));
        CUDA_CHECK(cudaMalloc((void**)&dpxy2, sizeof(double) * pointsSize));

        CUDA_CHECK(cudaMemcpy(dpx, px, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpy, py, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpxy2, pxy2, sizeof(double) * pointsSize, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&doutput, sizeof(bool) * pointsSize));
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        const Circumcircle circle(p1, p2, p3);
        const double coefPx = -2.0 * circle.x;
        const double coefPy = -2.0 * circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + 0.0000001;

        const int dim = min(max((int)ceil((sqrt((float)pointsSize)) / 32.0f) * 32, 32), 768);
        //cout << "DIM:   " << dim << endl;

        circlesShotgun<<<dim, dim>>>(doutput, dpx, dpy, dpxy2, coefPx, coefPy, comp, pointsSize, dim);
        CUDA_CHECK(cudaMemcpy(output, doutput, sizeof(bool) * pointsSize, cudaMemcpyDeviceToHost));
    }

    __host__ void cleanup() override {
        cudaFree(dpx);
        cudaFree(dpy);
        cudaFree(dpxy2);
        cudaFree(doutput);

        delete[] px;
        delete[] py;
        delete[] pxy2;
    }

    string getFileName() const override {
        return "kernel_shotgun";
    }

};
