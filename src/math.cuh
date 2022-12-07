#pragma once

#include <algorithm>
#include <vector>

#include "primitive.cuh"

using namespace std;

double det(double a, double b, double c, double d, double e, double f, double g, double h, double i){
    return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h);
}

double dets(double a, double b, double d, double e, double g, double h){
    return (a * e) + (b * g) + (d * h) - (e * g) - (b * d) - (a * h);
}

float distance(const Point& a, const Point& b){
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

bool isClockwise(Point a, Point b, Point c){
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

bool isAboveEdge(const Point& pEdge1, const Point& pEdge2, const Point& pTest){
    return ((double)pTest.x - (double)pEdge1.x) * ((double)pEdge2.y - (double)pEdge1.y) -
           ((double)pTest.y - (double)pEdge1.y) * ((double)pEdge2.x - (double)pEdge1.x) < 0.0;
}

bool isInCircle(const Point& a, const Point& b, const Point& c, const Point& d){
    // Fast "out" region test
    const int edgeSum = isAboveEdge(a, b, d) + isAboveEdge(b, c, d) + isAboveEdge(c, a, d);
    if(edgeSum == 0) return true;
    if(edgeSum >= 2) return false;

    return det(a.x - d.x, a.y - d.y, (a.x * a.x - d.x * d.x) + (a.y * a.y - d.y * d.y),
               b.x - d.x, b.y - d.y, (b.x * b.x - d.x * d.x) + (b.y * b.y - d.y * d.y),
               c.x - d.x, c.y - d.y, (c.x * c.x - d.x * d.x) + (c.y * c.y - d.y * d.y)) <= 0.0;
}

bool isInCircleSlow(const Point& a, const Point& b, const Point& c, const Point& d){
    return det(a.x - d.x, a.y - d.y, (a.x * a.x - d.x * d.x) + (a.y * a.y - d.y * d.y),
               b.x - d.x, b.y - d.y, (b.x * b.x - d.x * d.x) + (b.y * b.y - d.y * d.y),
               c.x - d.x, c.y - d.y, (c.x * c.x - d.x * d.x) + (c.y * c.y - d.y * d.y)) <= 0.0;
}