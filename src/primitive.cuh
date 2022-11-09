#pragma once

#include <algorithm>
#include <vector>

using namespace std;

// Szudzik's Function
int hashPair(int i1, int i2){
    return i1 >= i2 ? (i1 * i1 + i1 + i2) : (i1 + i2 * i2);
}

struct Point{
    float x, y;
};

struct Edge{
    int i1, i2;

    Edge reverse() const {
        return {i2, i1};
    }

    inline bool operator==(const Edge& edge) const {
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
    float xMin, xMid, xMax, yMin, yMid, yMax;

    Bounds(float xMinArg, float xMaxArg, float yMinArg, float yMaxArg){
        xMin = xMinArg;
        xMid = (xMinArg + xMaxArg) / 2.0f;
        xMax = xMaxArg;
        yMin = yMinArg;
        yMid = (yMinArg + yMaxArg) / 2.0f;
        yMax = yMaxArg;
    }
};

struct Wall{

    static Wall build(Bounds bounds, int depth){
        bool horizontal = depth % 2 == 0;
        return {horizontal, horizontal ? bounds.xMid : bounds.yMid};
    }

    const bool horizontal;
    const float locationDivider;

    float distance(Point point) const {
        return horizontal ? abs(locationDivider - point.x) : abs(locationDivider - point.y);
    }

    bool side(Point point) const {
        return horizontal ? point.x >= locationDivider : point.y >= locationDivider;
    }

    bool intersects(Point point1, Point point2) const {
        return side(point1) != side(point2);
    }
};