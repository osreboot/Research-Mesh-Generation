#pragma once

#include "math.cuh"

namespace circles_noop{

    inline void testSpeed(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        for(int i = 0; i < batchSize; i++){
            (void)0;
        }
    }

    inline vector<bool> testAccuracy(const vector<Point>& points, const Triangle& triangle, const int& offset, const int& batchSize){
        vector<bool> output;
        output.reserve(batchSize);
        for(int i = 0; i < batchSize; i++){
            output.push_back(isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[(offset + i) % points.size()]));
        }
        return output;
    }

}