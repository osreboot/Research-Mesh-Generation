

class Circles{

public:
    virtual void initialize(const Point *pointsArg, int pointsSizeArg) = 0;
    virtual void run(bool *output, const Point& p1, const Point& p2, const Point& p3) const = 0;
    virtual void cleanup() = 0;

    virtual string getFileName() const = 0;

};