#pragma once

#include "math.cuh"

namespace circles_serial{

    inline void testPureSpeed(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        for(int i = 0; i < batchSize; i++){
            isInCircleSlow(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]);
        }
    }

    inline vector<bool> testPureAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        vector<bool> output;
        for(int i = 0; i < batchSize; i++){
            output.push_back(isInCircleSlow(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]));
        }
        return output;
    }

    inline void testRegionsSpeed(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        for(int i = 0; i < batchSize; i++){
            isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]);
        }
    }

    inline vector<bool> testRegionsAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        vector<bool> output;
        for(int i = 0; i < batchSize; i++){
            output.push_back(isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]));
        }
        return output;
    }

    inline void testDistanceSpeed(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        const Circumcircle circle(points[triangle.i1], points[triangle.i2], points[triangle.i3]);
        for(int i = 0; i < batchSize; i++){
            circle.isInside(points[(offset + i) % points.size()]);
        }
    }

    inline vector<bool> testDistanceAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        vector<bool> output;
        const Circumcircle circle(points[triangle.i1], points[triangle.i2], points[triangle.i3]);
        for(int i = 0; i < batchSize; i++){
            output.push_back(circle.isInside(points[(offset + i) % points.size()]));
        }
        return output;
    }

}