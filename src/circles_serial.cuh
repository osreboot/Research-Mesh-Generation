#pragma once

#include "math.cuh"

namespace circles_serial{

    void testPureSpeed(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        for(int i = 0; i < batchSize; i++){
            isInCircleSlow(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]);
        }
    }

    vector<bool> testPureAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        vector<bool> output;
        for(int i = 0; i < batchSize; i++){
            output.push_back(isInCircleSlow(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]));
        }
        return output;
    }

    void testRegionsSpeed(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        for(int i = 0; i < batchSize; i++){
            isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]);
        }
    }

    vector<bool> testRegionsAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        vector<bool> output;
        for(int i = 0; i < batchSize; i++){
            output.push_back(isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]));
        }
        return output;
    }

}