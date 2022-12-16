

class CirclesLocal{

public:
    virtual __host__ __device__ void initialize(const double *pxArg, const double *pyArg, int pointsSizeArg) = 0;
    virtual __host__ __device__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const = 0;
    virtual __host__ __device__ void cleanup() = 0;

};