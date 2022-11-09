#pragma once

#include "math.cuh"

namespace circles_linear{

    void circles(const vector<Point>& points, const int& a){
        for(int i = 0; i < points.size(); i++){
            if(i < a || i > a + 2){
                isInCircle(points[a], points[a + 1], points[a + 2], points[i]);
            }
        }
    }

}