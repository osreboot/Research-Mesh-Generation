#include "math.cuh"

class CirclesLocalSerialPure : public CirclesLocal{

private:
    const double *px = nullptr, *py = nullptr;
    int pointsSize = 0;

public:
    __host__ __device__ void initialize(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;
    }

    __host__ __device__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        for(int i = 0; i < pointsSize; i++){
            output[i] = isInCircle(p1, p2, p3, {px[i], py[i]});
        }
    }

    __host__ __device__ void cleanup() override {}

};
