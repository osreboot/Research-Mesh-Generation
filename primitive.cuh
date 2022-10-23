#pragma once

#define PARTITION_DIMENSION 20

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

class Partition{

public:
    vector<int> partition[PARTITION_DIMENSION][PARTITION_DIMENSION];

    void insert(Point point, int index){
        const int xPart = floor(point.x * PARTITION_DIMENSION);
        const int yPart = floor(point.y * PARTITION_DIMENSION);
        partition[xPart][yPart].push_back(index);
    }

    vector<int> search(float x, float y, float r, Bounds bounds) const {
        return search(x - r, x + r, y - r, y + r, bounds);
    }

    vector<int> search(float xMin, float xMax, float yMin, float yMax, Bounds bounds) const {
        vector<int> output;
        int boundXMin = max(min((int)floor(bounds.xMin * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int boundXMax = max(min((int)floor(bounds.xMax * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int boundYMin = max(min((int)floor(bounds.yMin * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int boundYMax = max(min((int)floor(bounds.yMax * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        xMin = max(min((int)floor(xMin * PARTITION_DIMENSION), boundXMax), boundXMin);
        xMax = max(min((int)floor(xMax * PARTITION_DIMENSION), boundXMax), boundXMin);
        yMin = max(min((int)floor(yMin * PARTITION_DIMENSION), boundYMax), boundYMin);
        yMax = max(min((int)floor(yMax * PARTITION_DIMENSION), boundYMax), boundYMin);
        //cout << "    " << xMax - xMin << " | " << yMax - yMin << endl;
        for(int x = xMin; x <= xMax; x++){
            for(int y = yMin; y <= yMax; y++){
                output.insert(output.begin(), partition[x][y].begin(), partition[x][y].end());
            }
        }
        return output;
    }

    vector<int> search(float xMid, float yMid, int offset, Bounds bounds) const {
        vector<int> output;
        int boundXMin = max(min((int)floor(bounds.xMin * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int boundXMax = max(min((int)ceil(bounds.xMax * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int boundYMin = max(min((int)floor(bounds.yMin * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int boundYMax = max(min((int)ceil(bounds.yMax * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        int xMin = max(min((int)floor(xMid * PARTITION_DIMENSION) - offset, boundXMax), boundXMin);
        int xMax = max(min((int)floor(xMid * PARTITION_DIMENSION) + offset, boundXMax), boundXMin);
        int yMin = max(min((int)floor(yMid * PARTITION_DIMENSION) - offset, boundYMax), boundYMin);
        int yMax = max(min((int)floor(yMid * PARTITION_DIMENSION) + offset, boundYMax), boundYMin);
        //cout << "[" << offset << "] " << xMax - xMin << " | " << yMax - yMin << endl;
        if(offset == 0){
            output.insert(output.begin(), partition[xMin][yMin].begin(), partition[xMin][yMin].end());
        }else{
            for(int x = xMin; x <= xMax; x++){
                output.insert(output.begin(), partition[x][yMin].begin(), partition[x][yMin].end());
                output.insert(output.begin(), partition[x][yMax].begin(), partition[x][yMax].end());
            }
            for(int y = yMin + 1; y < yMax; y++){
                output.insert(output.begin(), partition[xMin][y].begin(), partition[xMin][y].end());
                output.insert(output.begin(), partition[xMax][y].begin(), partition[xMax][y].end());
            }
        }
        return output;
    }

};