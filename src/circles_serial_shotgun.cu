#include "math.cuh"

class CirclesSerialShotgun : public Circles{

private:
    const double *px = nullptr, *py = nullptr;
    double *pxy2 = nullptr;
    int pointsSize = 0;

public:
    void initialize(const double *pxArg, const double *pyArg, int pointsSizeArg) override {
        px = pxArg;
        py = pyArg;
        pointsSize = pointsSizeArg;

        pxy2 = new double[pointsSizeArg];

        for(int i = 0; i < pointsSize; i++){
            pxy2[i] = px[i] * px[i] + py[i] * py[i];
        }
    }

    void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        const Circumcircle circle(p1, p2, p3);
        const double coefPx = -2.0 * circle.x;
        const double coefPy = -2.0 * circle.y;
        const double comp = circle.r2 - circle.x * circle.x - circle.y * circle.y + 0.0000001;

        for(int i = 0; i < pointsSize; i++){
            output[i] = coefPx * px[i] + coefPy * py[i] + pxy2[i] <= comp;
        }
    }

    void cleanup() override {
        delete[] pxy2;
    }

    string getFileName() const override {
        return "serial_shotgun";
    }

};
