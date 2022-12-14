#pragma once

#include <algorithm>
#include <vector>

using namespace std;

// Szudzik's Function
int hashPair(int i1, int i2){
    return i1 >= i2 ? (i1 * i1 + i1 + i2) : (i1 + i2 * i2);
}

struct Point{
    double x, y;
};

ostream& operator<<(ostream &ostr, const Point& p){
    return ostr << "(" << p.x << "," << p.y << ")";
}

struct Edge{
    int i1, i2;

    __host__ __device__ Edge reverse() const {
        return {i2, i1};
    }

    __host__ __device__ __inline__ bool operator==(const Edge& edge) const {
        return (i1 == edge.i1 && i2 == edge.i2);
    }
};

template<>
struct hash<Edge>{
    inline size_t operator()(const Edge& edge) const {
        return ::hashPair(edge.i1, edge.i2);
    }
};

struct Triangle{
    int i1, i2, i3;

    inline bool operator==(const Triangle& triangle) const {
        return (i1 == triangle.i1 && i2 == triangle.i2 && i3 == triangle.i3);
    }
};

template<>
struct hash<Triangle>{
    inline size_t operator()(const Triangle& triangle) const {
        return ::hashPair(::hashPair(triangle.i1, triangle.i2), triangle.i3);
    }
};

class Bounds{

public:
    double xMin, xMid, xMax, yMin, yMid, yMax;

    __host__ __device__ Bounds(double xMinArg, double xMaxArg, double yMinArg, double yMaxArg){
        xMin = xMinArg;
        xMid = (xMinArg + xMaxArg) / 2.0f;
        xMax = xMaxArg;
        yMin = yMinArg;
        yMid = (yMinArg + yMaxArg) / 2.0f;
        yMax = yMaxArg;
    }

    __host__ __device__ bool contains(const Point& point) const {
        return point.x >= xMin && point.x <= xMax && point.y >= yMin && point.y <= yMax;
    }
};

struct Wall{

    __host__ __device__ static Wall build(const Bounds& bounds, const int& depth){
        bool horizontal = depth % 2 == 0;
        return {horizontal, horizontal ? bounds.xMid : bounds.yMid};
    }

    const bool horizontal;
    const double locationDivider;

    double distance(const Point& point) const {
        return horizontal ? abs(locationDivider - point.x) : abs(locationDivider - point.y);
    }

    __host__ __device__ double distance(const double px, const double py) const {
        return horizontal ? abs(locationDivider - px) : abs(locationDivider - py);
    }

    bool side(const Point& point) const {
        return horizontal ? point.x >= locationDivider : point.y >= locationDivider;
    }

    __host__ __device__ bool side(const double px, const double py) const {
        return horizontal ? px >= locationDivider : py >= locationDivider;
    }

    bool intersects(const Point& point1, const Point& point2) const {
        return side(point1) != side(point2);
    }

    __host__ __device__ bool intersects(const double px1, const double py1, const double px2, const double py2) const {
        return side(px1, py1) != side(px2, py2);
    }
};