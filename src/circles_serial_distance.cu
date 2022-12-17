#include "math.cuh"

class CirclesSerialDistance : public Circles{

private:
    const double *px = nullptr, *py = nullptr;
    int pointsSize = 0;

public:
    __host__ void load(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;
    }

    __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) override {
        const Circumcircle circle(p1, p2, p3);
        for(int i = 0; i < pointsSize; i++){
            output[i] = circle.isInside(px[i], py[i]);
        }
    }

    __host__ void save(bool *output) override {}

    __host__ void cleanup() override {}

    string getFileName() const override {
        return "serial_distance";
    }

};
