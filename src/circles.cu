

class Circles{

public:
    virtual __host__ void load(const double *pxArg, const double *pyArg, int pointsSizeArg) = 0;
    virtual __host__ void run(bool *output, const Point& p1, const Point& p2, const Point& p3) = 0;
    virtual __host__ void save(bool *output) = 0;
    virtual __host__ void cleanup() = 0;

    virtual string getFileName() const = 0;

};