#include "math.cuh"

class CirclesSerialDisjoint : public Circles{

private:
    const Point* points = nullptr;
    int pointsSize = 0;

public:
    void initialize(const Point *pointsArg, int pointsSizeArg) override {
        points = pointsArg;
        pointsSize = pointsSizeArg;
    }

    void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const override {
        /*

         need to compute:

         pxy2 = p.x * p.x + p.y * p.y

         a = a.x - p.x
         b = a.y - p.y
         c = axy2 - pxy2
         d = b.x - p.x
         e = b.y - p.y
         f = bxy2 - pxy2
         g = c.y - p.y
         h = c.y - p.y
         i = cxy2 - pxy2



         */

        for(int i = 0; i < pointsSize; i++){
            output[i] = isInCircle(p1, p2, p3, points[i]);
        }
    }

    void cleanup() override {}

    string getFileName() const override {
        return "serial_pure";
    }

};
