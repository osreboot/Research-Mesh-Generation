#include "math.cuh"

class CirclesNoop : public Circles{

private:
    const Point* points = nullptr;
    int pointsSize = 0;

public:
    void initialize(const Point *pointsArg, int pointsSizeArg) override {
        points = pointsArg;
        pointsSize = pointsSizeArg;
    }

    void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        for(int i = 0; i < pointsSize; i++){
            output[i] = false;
        }
    }

    void cleanup() override {}

    string getFileName() const override {
        return "serial_noop";
    }

};