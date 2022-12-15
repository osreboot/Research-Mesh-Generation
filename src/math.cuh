#pragma once

#include <algorithm>
#include <vector>

#include "primitive.cuh"

using namespace std;

__host__ __device__ inline double det(double a, double b, double c, double d, double e, double f, double g, double h, double i){
    return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h);
}

__host__ __device__ inline double dets(double a, double b, double d, double e, double g, double h){
    return (a * e) + (b * g) + (d * h) - (e * g) - (b * d) - (a * h);
}

__host__ __device__ inline double distance(const Point& a, const Point& b){
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__host__ __device__ inline double distance(const double px1, const double py1, const double px2, const double py2){
    return sqrt((px1 - px2) * (px1 - px2) + (py1 - py2) * (py1 - py2));
}

inline bool isClockwise(Point a, Point b, Point c){
    return 1.0 * a.x * b.y + 1.0 * a.y * c.x + 1.0 * b.x * c.y -
           1.0 * b.y * c.x - 1.0 * a.y * b.x - 1.0 * a.x * c.y < 0.0;
}

Triangle makeClockwise(const vector<Point>& points, Triangle triangle){
    if(!isClockwise(points[triangle.i1], points[triangle.i2], points[triangle.i3])) {
        int temp = triangle.i2;
        triangle.i2 = triangle.i3;
        triangle.i3 = temp;
    }
    return triangle;
}

Triangle makeSequential(const vector<Point>& points, const Triangle& triangle){
    int indexMin = min(min(triangle.i1, triangle.i2), triangle.i3);
    if(indexMin == triangle.i1) return triangle;
    if(indexMin == triangle.i2) return {triangle.i2, triangle.i3, triangle.i1};
    else return {triangle.i3, triangle.i1, triangle.i2};
}

inline bool isAboveEdge(const Point& pEdge1, const Point& pEdge2, const Point& pTest){
    return ((double)pTest.x - (double)pEdge1.x) * ((double)pEdge2.y - (double)pEdge1.y) -
           ((double)pTest.y - (double)pEdge1.y) * ((double)pEdge2.x - (double)pEdge1.x) < 0.0;
}

inline bool isAboveEdge(const double pxEdge1, const double pyEdge1,
                        const double pxEdge2, const double pyEdge2,
                        const double pxTest, const double pyTest){
    return ((double)pxTest - (double)pxEdge1) * ((double)pyEdge2 - (double)pyEdge1) -
           ((double)pyTest - (double)pyEdge1) * ((double)pxEdge2 - (double)pxEdge1) < 0.0;
}

inline bool isInCircle(const Point& a, const Point& b, const Point& c, const Point& d){
    return det(a.x - d.x, a.y - d.y, (a.x * a.x - d.x * d.x) + (a.y * a.y - d.y * d.y),
               b.x - d.x, b.y - d.y, (b.x * b.x - d.x * d.x) + (b.y * b.y - d.y * d.y),
               c.x - d.x, c.y - d.y, (c.x * c.x - d.x * d.x) + (c.y * c.y - d.y * d.y)) <= 0.0;
}

bool isInCircleRegions(const Point& a, const Point& b, const Point& c, const Point& d){
    // Fast "out" region test
    //   *Based on tests conducted, this is actually slightly slower than the standard method based on the general test
    //    case.
    const int edgeSum = isAboveEdge(a, b, d) + isAboveEdge(b, c, d) + isAboveEdge(c, a, d);
    if(edgeSum == 0) return true;
    if(edgeSum >= 2) return false;

    return det(a.x - d.x, a.y - d.y, (a.x * a.x - d.x * d.x) + (a.y * a.y - d.y * d.y),
               b.x - d.x, b.y - d.y, (b.x * b.x - d.x * d.x) + (b.y * b.y - d.y * d.y),
               c.x - d.x, c.y - d.y, (c.x * c.x - d.x * d.x) + (c.y * c.y - d.y * d.y)) <= 0.0;
}

__host__ __device__ __inline__ bool isInside(const Point& point, const double x, const double y, const double r2){
    return ((x - point.x) * (x - point.x) + (y - point.y) * (y - point.y)) <= r2 + 0.0000001;
}

class Circumcircle{

public:
    double x, y, r2;

    __host__ __device__ Circumcircle(const Point& p1, const Point& p2, const Point& p3){
        // Algorithm source: https://mathworld.wolfram.com/Circumcircle.html
        double a = dets(p1.x, p1.y,
                        p2.x, p2.y,
                        p3.x, p3.y);

        double bx = -dets(p1.x * p1.x + p1.y * p1.y, p1.y,
                          p2.x * p2.x + p2.y * p2.y, p2.y,
                          p3.x * p3.x + p3.y * p3.y, p3.y);
        double by = dets(p1.x * p1.x + p1.y * p1.y, p1.x,
                         p2.x * p2.x + p2.y * p2.y, p2.x,
                         p3.x * p3.x + p3.y * p3.y, p3.x);

        double c = -det(p1.x * p1.x + p1.y * p1.y, p1.x, p1.y,
                        p2.x * p2.x + p2.y * p2.y, p2.x, p2.y,
                        p3.x * p3.x + p3.y * p3.y, p3.x, p3.y);

        x = -bx / (2.0 * a);
        y = -by / (2.0 * a);

        r2 = sqrt(bx * bx + by * by - 4.0 * a * c) / (2.0 * abs(a));
        r2 *= r2;
    }

    __host__ __device__ __inline__ bool isInside(const Point& point) const {
        return ((x - point.x) * (x - point.x) + (y - point.y) * (y - point.y)) <= r2 + 0.000000000001;
    }

    __host__ __device__ __inline__ bool isInside(const double px, const double py) const {
        return ((x - px) * (x - px) + (y - py) * (y - py)) <= r2 + 0.000000000001;
    }
};