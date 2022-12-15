#include "math.cuh"

class CirclesSerialRegions : public Circles{

private:
    const double *px = nullptr, *py = nullptr;
    int pointsSize = 0;

public:
    void initialize(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;
    }

    void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        for(int i = 0; i < pointsSize; i++){
            output[i] = isInCircleRegions(p1, p2, p3, {px[i], py[i]});
        }
    }

    void cleanup() override {}

    string getFileName() const override {
        return "serial_regions";
    }

};
