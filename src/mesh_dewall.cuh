#pragma once

#include <unordered_set>

#include "primitive.cuh"
#include "math.cuh"
#include "profiler.cuh"

#define PARTITION_DIMENSION 100
#define CELL_SIZE (1.0f / (float)PARTITION_DIMENSION)

using namespace std;

namespace mesh_dewall{

    class PartitionBounds{

    public:
        int xMin, xMax, yMin, yMax;

        PartitionBounds(float xMinArg, float xMaxArg, float yMinArg, float yMaxArg){
            xMin = max(min((int)floor(xMinArg * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
            xMax = max(min((int)floor(xMaxArg * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
            yMin = max(min((int)floor(yMinArg * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
            yMax = max(min((int)floor(yMaxArg * PARTITION_DIMENSION), PARTITION_DIMENSION - 1), 0);
        }

        void constrain(float xMinArg, float xMaxArg, float yMinArg, float yMaxArg){
            xMin = max(min((int)floor(xMinArg * PARTITION_DIMENSION), xMax), xMin);
            xMax = max(min((int)floor(xMaxArg * PARTITION_DIMENSION), xMax), xMin);
            yMin = max(min((int)floor(yMinArg * PARTITION_DIMENSION), yMax), yMin);
            yMax = max(min((int)floor(yMaxArg * PARTITION_DIMENSION), yMax), yMin);
        }

        void constrain(const Bounds& bounds){
            constrain(bounds.xMin, bounds.xMax, bounds.yMin, bounds.yMax);
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

    };

    PartitionBounds boundsCircumcircle(const Point& p1, const Point& p2, const Point& p3){
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

        double x0 = -bx / (2.0 * a);
        double y0 = -by / (2.0 * a);

        double r = sqrt(bx * bx + by * by - 4.0 * a * c) / (2.0 * abs(a));

        return PartitionBounds(x0 - r, x0 + r, y0 - r, y0 + r);
    }

    void testPoints(const vector<Point>& points, const vector<int>& indices, const Edge& edge, int& index){
        for(int i : indices){
            if(isAboveEdge(points[edge.i1], points[edge.i2], points[i])) {
                if(index == -1) {
                    index = i;
                }else {
                    if(isInCircle(points[edge.i2], points[edge.i1], points[index], points[i])){
                        index = i;
                    }
                }
            }
        }
    }

    // Attempt to find a point to pair with the supplied Edge to complete a triangle
    int findPoint(const vector<Point>& points, const Partition& partition, int depth, const Bounds& bounds,
                  const Edge &edgeActive){
        float edgeCenterX = (points[edgeActive.i1].x + points[edgeActive.i2].x) / 2.0f;
        float edgeCenterY = (points[edgeActive.i1].y + points[edgeActive.i2].y) / 2.0f;

        // Traverse the spatial partition of points and attempt to find at least one close, valid point
        profiler::startSection(depth, profiler::L_FIRST);
        int indexP3 = -1;
        PartitionBounds partitionBounds(0.0f, 0.0f, 0.0f, 0.0f);
        for(float offset = 0.0f; offset <= 1.0f; offset += CELL_SIZE){
            partitionBounds = PartitionBounds(edgeCenterX - offset, edgeCenterX + offset,
                                              edgeCenterY - offset, edgeCenterY + offset);
            partitionBounds.constrain(bounds);

            // TODO maybe start with 3x3 partition cells for a better initial circle
            if(offset == 0.0f){
                testPoints(points, partition.partition[partitionBounds.xMin][partitionBounds.yMin], edgeActive, indexP3);
            }else{
                for(int x = partitionBounds.xMin; x <= partitionBounds.xMax; x++){
                    testPoints(points, partition.partition[x][partitionBounds.yMin], edgeActive, indexP3);
                    testPoints(points, partition.partition[x][partitionBounds.yMax], edgeActive, indexP3);
                }
                for(int y = partitionBounds.yMin + 1; y < partitionBounds.yMax; y++){
                    testPoints(points, partition.partition[partitionBounds.xMin][y], edgeActive, indexP3);
                    testPoints(points, partition.partition[partitionBounds.xMax][y], edgeActive, indexP3);
                }
            }

            if(indexP3 != -1) break;
        }
        profiler::stopSection(depth, profiler::L_FIRST);

        // Optimize the existing triangle by searching for better matches inside its circumcircle
        if(indexP3 != -1){
            profiler::startSection(depth, profiler::L_CIRCLE);

            PartitionBounds partitionBounds2 = boundsCircumcircle(points[edgeActive.i2], points[edgeActive.i1], points[indexP3]);
            partitionBounds2.constrain(bounds);

            // Omit already searched cells
            for(int x = partitionBounds2.xMin; x <= partitionBounds2.xMax; x++){
                for(int y = partitionBounds2.yMin; y <= partitionBounds2.yMax; y++){
                    if(x >= partitionBounds.xMin && x <= partitionBounds.xMax &&
                       y >= partitionBounds.yMin && y <= partitionBounds.yMax){
                        y = partitionBounds.yMax;
                    }else testPoints(points, partition.partition[x][y], edgeActive, indexP3);
                }
            }
            profiler::stopSection(depth, profiler::L_CIRCLE);
        }

        return indexP3;
    }

    // Algorithm source: https://doi.org/10.1016/S0010-4485(97)00082-1
    vector<Triangle> triangulateRecursive(const vector<Point>& points, const vector<int>& indices,
                                          const Partition& partition, Bounds bounds, unordered_set<Edge> edgesActive,
                                          int depth){
        profiler::startBranch(depth);

        // Calculate dividing wall
        profiler::startSection(depth, profiler::WALL);
        Wall wall = Wall::build(bounds, depth);
        profiler::stopSection(depth, profiler::WALL);
        profiler::startSection(depth, profiler::INIT);

        // Divide points by wall side
        vector<int> indicesLeft, indicesRight;
        for(int i : indices){
            if(wall.side(points[i])) indicesRight.push_back(i);
            else indicesLeft.push_back(i);
        }

        vector<Triangle> output;
        unordered_set<Edge> edgesActiveWall, edgesActive1, edgesActive2;

        // Initialize with first edges if we have no active edges
        if(edgesActive.empty() && depth == 0){
            // Find nearest point to dividing line (first edge point)
            int aIndex = indices[0];
            float aDistance = wall.distance(points[indices[0]]);
            for(int i : indices){
                float iDistance = wall.distance(points[i]);
                if(iDistance < aDistance){
                    aIndex = i;
                    aDistance = iDistance;
                }
            }
            bool aSide = wall.side(points[aIndex]);

            // Find nearest point to A on other side of dividing wall (second edge point)
            int bIndex = aSide ? indicesLeft[0] : indicesRight[0];
            float bDistance = distance(points[aIndex], points[bIndex]);
            for(int i : (aSide ? indicesLeft : indicesRight)){
                float iDistance = distance(points[aIndex], points[i]);
                if(iDistance < bDistance){
                    bIndex = i;
                    bDistance = iDistance;
                }
            }

            edgesActive.insert({aIndex, bIndex});
            edgesActive.insert({bIndex, aIndex});
        }

        // Divide inherited active edges
        for(Edge edge : edgesActive){
            if(wall.intersects(points[edge.i1], points[edge.i2])){
                edgesActiveWall.insert(edge);
            }else if(wall.side(points[edge.i1])){
                edgesActive2.insert(edge);
            }else edgesActive1.insert(edge);
        }

        profiler::stopSection(depth, profiler::INIT);

        // For all active edges, attempt to complete a triangle and update the active edges list
        while(!edgesActiveWall.empty()){
            Edge edge = *edgesActiveWall.begin();
            edgesActiveWall.erase(edgesActiveWall.begin());

            int indexP3 = findPoint(points, partition, depth, bounds, edge);

            if(indexP3 > -1){ // Check if we've made a new triangle
                // Add new triangle to output list
                //    This needs to be sequential so that we can safely handle potential duplicate output triangles
                profiler::startSection(depth, profiler::SAVE);
                output.push_back({edge.i1, indexP3, edge.i2});
                profiler::stopSection(depth, profiler::SAVE);

                // Update active edges based on the new triangle
                profiler::startSection(depth, profiler::CHAIN);
                for(Edge e : vector<Edge>{{indexP3, edge.i2}, {edge.i1, indexP3}}){
                    if(wall.intersects(points[e.i1], points[e.i2])){
                        if(edgesActiveWall.count(e) || edgesActiveWall.count(e.reverse())){
                            edgesActiveWall.erase(e);
                            edgesActiveWall.erase(e.reverse());
                        }else edgesActiveWall.insert(e);
                    }else if(wall.side(points[e.i1])){
                        if(edgesActive2.count(e) || edgesActive2.count(e.reverse())){
                            edgesActive2.erase(e);
                            edgesActive2.erase(e.reverse());
                        }else edgesActive2.insert(e);
                    }else{
                        if(edgesActive1.count(e) || edgesActive1.count(e.reverse())){
                            edgesActive1.erase(e);
                            edgesActive1.erase(e.reverse());
                        }else edgesActive1.insert(e);
                    }
                }
                profiler::stopSection(depth, profiler::CHAIN);
            }
        }

        profiler::stopBranch(depth);

        // Recursively call delaunay triangulation on both sides of the wall
        if(!edgesActive1.empty()){
            Bounds bounds1(bounds.xMin, wall.horizontal ? bounds.xMid : bounds.xMax, bounds.yMin, (!wall.horizontal) ? bounds.yMid : bounds.yMax);
            vector<Triangle> triangles = triangulateRecursive(points, indicesLeft, partition, bounds1, edgesActive1, depth + 1);
            profiler::startBranch(depth);
            profiler::startSection(depth, profiler::MERGE);
            output.insert(output.begin(), triangles.begin(), triangles.end());
            profiler::stopSection(depth, profiler::MERGE);
            profiler::stopBranch(depth);
        }
        if(!edgesActive2.empty()){
            Bounds bounds2(wall.horizontal ? bounds.xMid : bounds.xMin, bounds.xMax, (!wall.horizontal) ? bounds.yMid : bounds.yMin, bounds.yMax);
            vector<Triangle> triangles = triangulateRecursive(points, indicesRight, partition, bounds2, edgesActive2, depth + 1);
            profiler::startBranch(depth);
            profiler::startSection(depth, profiler::MERGE);
            output.insert(output.begin(), triangles.begin(), triangles.end());
            profiler::stopSection(depth, profiler::MERGE);
            profiler::stopBranch(depth);
        }

        return output;
    }

    vector<Triangle> triangulate(const vector<Point>& points, const vector<int>& indices){
        Partition partition;
        for(int i : indices) partition.insert(points[i], i);

        return triangulateRecursive(points, indices, partition, Bounds(0.0f, 1.0f, 0.0f, 1.0f), {}, 0);
    }

}