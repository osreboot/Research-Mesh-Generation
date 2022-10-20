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

float distance(Point a, Point b){
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

Triangle makeSequential(const vector<Point>& points, Triangle triangle){
    int indexMin = min(min(triangle.i1, triangle.i2), triangle.i3);
    int indexMax = max(max(triangle.i1, triangle.i2), triangle.i3);
    int indexMid = triangle.i1 + triangle.i2 + triangle.i3 - indexMin - indexMax;
    return makeClockwise(points, {indexMin, indexMid, indexMax});
}

bool isAboveEdge(const vector<Point>& points, Edge edge, int indexPoint){
    return 1.0 * (points[indexPoint].x - points[edge.i1].x) * (points[edge.i2].y - points[edge.i1].y) -
           1.0 * (points[indexPoint].y - points[edge.i1].y) * (points[edge.i2].x - points[edge.i1].x) < 0.0;
}

bool isInCircle(Point a, Point b, Point c, Point d){
    // Fast "out" region test
    const vector<Point> points = {a, b, c, d};
    if(isAboveEdge(points, {0, 1}, 3) + isAboveEdge(points, {1, 2}, 3) +
       isAboveEdge(points, {2, 0}, 3) >= 2) return false;

    return det(a.x - d.x, a.y - d.y, (a.x * a.x - d.x * d.x) + (a.y * a.y - d.y * d.y),
               b.x - d.x, b.y - d.y, (b.x * b.x - d.x * d.x) + (b.y * b.y - d.y * d.y),
               c.x - d.x, c.y - d.y, (c.x * c.x - d.x * d.x) + (c.y * c.y - d.y * d.y)) < 0.0;
}