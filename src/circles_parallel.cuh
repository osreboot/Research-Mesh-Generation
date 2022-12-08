#pragma once

#include "math.cuh"

namespace circles_parallel{

    __device__ double det(double a, double b, double c, double d, double e, double f, double g, double h, double i){
        return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h);
    }

    __global__ void circles(bool *out, const double* __restrict__ px, const double* __restrict__ py, double ax, double ay, double bx, double by, double cx, double cy, int batchSize, int threads){
        for(int i = threadIdx.x; i < batchSize; i += threads){
            out[i] = det(ax - px[i], ay - py[i], (ax * ax - px[i] * px[i]) + (ay * ay - py[i] * py[i]),
                         bx - px[i], by - py[i], (bx * bx - px[i] * px[i]) + (by * by - py[i] * py[i]),
                         cx - px[i], cy - py[i], (cx * cx - px[i] * px[i]) + (cy * cy - py[i] * py[i])) <= 0.0;
        }
    }

    __host__ vector<bool> testAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        bool *out, *dout;
        double *px, *py, *dpx, *dpy;

        px = (double*)malloc(sizeof(double) * batchSize);
        py = (double*)malloc(sizeof(double) * batchSize);
        out = (bool*)malloc(sizeof(bool) * batchSize);

        for(int i = offset; i < offset + batchSize; i++){
            px[i - offset] = points[i % points.size()].x;
            py[i - offset] = points[i % points.size()].y;
        }

        cudaMalloc((void**)&dpx, sizeof(double) * batchSize);
        cudaMalloc((void**)&dpy, sizeof(double) * batchSize);
        cudaMalloc((void**)&dout, sizeof(bool) * batchSize);
        cudaMemcpy(dpx, px, sizeof(double) * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(dpy, py, sizeof(double) * batchSize, cudaMemcpyHostToDevice);

        const int threads = min(batchSize, 20);

        circles<<<1,threads>>>(dout, dpx, dpy,
                               points[triangle.i1].x, points[triangle.i1].y,
                               points[triangle.i2].x, points[triangle.i2].y,
                               points[triangle.i3].x, points[triangle.i3].y, batchSize, threads);

        cudaDeviceSynchronize();

        cudaMemcpy(out, dout, sizeof(bool) * batchSize, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        vector<bool> output;
        output.reserve(batchSize);
        for(int i = 0; i < batchSize; i++){
            output.push_back(out[i]);
        }

        cudaFree(dpx);
        cudaFree(dpy);
        cudaFree(dout);
        free(px);
        free(py);

        return output;
    }

    __host__ void testSpeed(const vector<Point>& points, const Triangle& triangle, int offset, int batchSize){
        testAccuracy(points, triangle, offset, batchSize);
    }

}