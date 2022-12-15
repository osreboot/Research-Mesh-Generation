#include "profiler_mesh.cuh"

class Mesh{

public:
    virtual void run(Triangle *connections, int &connectionsSize, const double *px, const double *py, int pointsSize) const = 0;

    virtual string getFileName() const = 0;
    virtual vector<profiler_mesh::Section> getProfilerSections() const = 0;

};