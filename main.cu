#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "primitive.cuh"
#include "math.cuh"
#include "profiler.cuh"

using namespace std;

vector<int> delaunaySearchCircumcircle(const vector<Point>& points, const vector<int>& indices, const Partition& partition, Bounds bounds, Edge edgeActive, int indexP3){
    Triangle triangle = makeClockwise(points,{edgeActive.i1, edgeActive.i2, indexP3});
    const Point p1 = points[triangle.i1];
    const Point p2 = points[triangle.i2];
    const Point p3 = points[triangle.i3];

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

    return partition.search((float)x0, (float)y0, (float)r, bounds);
}

int delaunayFindPoint(const vector<Point>& points, const vector<int>& indices, const Partition& partition, int depth, Bounds bounds, Edge edgeActive){
    //float edgeDistance = distance(points[edgeActive.i1], points[edgeActive.i2]);
    float edgeCenterX = (points[edgeActive.i1].x + points[edgeActive.i2].x) / 2.0f;
    float edgeCenterY = (points[edgeActive.i1].y + points[edgeActive.i2].y) / 2.0f;

    profiler::startSection(depth, profiler::L_FIRST);
    int indexP3 = -1;
    for(int offset = 0; offset <= PARTITION_DIMENSION; offset++){
        for(int i : partition.search(edgeCenterX, edgeCenterY, offset, bounds)){
            if(isAboveEdge(points, edgeActive, i)){ // TODO move this test into partition
                if(indexP3 == -1){
                    indexP3 = i;
                }else{
                    Triangle triangle = makeClockwise(points, {edgeActive.i1, edgeActive.i2, indexP3});
                    if(isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[i])){
                        indexP3 = i;
                    }
                }
            }
        }
        if(indexP3 != -1) break;
    }
    profiler::stopSection(depth, profiler::L_FIRST);

    if(indexP3 != -1){
        profiler::startSection(depth, profiler::L_CIRCLE);
        bool foundNewPoint;
       // do{
            foundNewPoint = false;
            for(int i : delaunaySearchCircumcircle(points, indices, partition, bounds, edgeActive, indexP3)){
                if(isAboveEdge(points, edgeActive, i)){
                    Triangle triangle = makeClockwise(points, {edgeActive.i1, edgeActive.i2, indexP3});
                    if(isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[i])){
                        indexP3 = i;
                        foundNewPoint = true;
                    }
                }
            }
       // }while(foundNewPoint);
        profiler::stopSection(depth, profiler::L_CIRCLE);
    }

    return indexP3;

    /*
    int indexP3 = -1;
    for(int offset = 0; offset <= PARTITION_DIMENSION; offset++){
        for(int i : partition.search(edgeCenterX, edgeCenterY, offset, bounds)){
            if(isAboveEdge(points, edgeActive, i)){
                if(indexP3 == -1){
                    indexP3 = i;
                }else{
                    Triangle triangle = makeClockwise(points, {edgeActive.i1, edgeActive.i2, indexP3});
                    if(isInCircle(points[triangle.i1], points[triangle.i2], points[triangle.i3], points[i])){
                        indexP3 = i;
                    }
                }
            }
        }
    }*/
    return indexP3;
}

// Algorithm source: https://doi.org/10.1016/S0010-4485(97)00082-1
unordered_set<Triangle> delaunay(const vector<Point>& points, const vector<int>& indices, const Partition& partition, Bounds bounds,
                                 unordered_set<Edge> edgesActive, int depth){

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

    unordered_set<Triangle> output;
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

        int p3Index = delaunayFindPoint(points, indices, partition, depth, bounds, edge);

        if(p3Index > -1){ // Check if we've made a new triangle
            profiler::startSection(depth, profiler::SAVE);
            Triangle triangle = makeClockwise(points, {p3Index, edge.i1, edge.i2});
            Triangle triangleOutput = makeSequential(points, triangle);
            if(output.count(triangleOutput)) cerr << "Generated a duplicate triangle!" << endl;
            output.insert(triangleOutput);
            profiler::stopSection(depth, profiler::SAVE);

            // Update active edges based on the new triangle
            profiler::startSection(depth, profiler::CHAIN);
            for(Edge e : vector<Edge>{{triangle.i1, triangle.i2}, {triangle.i3, triangle.i1}}){
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
                }else {
                    if (edgesActive1.count(e) || edgesActive1.count(e.reverse())) {
                        edgesActive1.erase(e);
                        edgesActive1.erase(e.reverse());
                    } else edgesActive1.insert(e);
                }
            }
            profiler::stopSection(depth, profiler::CHAIN);
        }
    }

    profiler::stopBranch(depth);

    // Recursively call delaunay triangulation on both sides of the wall
    if(!edgesActive1.empty()){
        Bounds bounds1(bounds.xMin, wall.horizontal ? bounds.xMid : bounds.xMax, bounds.yMin, (!wall.horizontal) ? bounds.yMid : bounds.yMax);
        unordered_set<Triangle> triangles = delaunay(points, indicesLeft, partition, bounds1, edgesActive1, depth + 1);
        profiler::startBranch(depth);
        profiler::startSection(depth, profiler::MERGE);
        output.insert(triangles.begin(), triangles.end());
        profiler::stopSection(depth, profiler::MERGE);
        profiler::stopBranch(depth);
    }
    if(!edgesActive2.empty()){
        Bounds bounds2(wall.horizontal ? bounds.xMid : bounds.xMin, bounds.xMax, (!wall.horizontal) ? bounds.yMid : bounds.yMin, bounds.yMax);
        unordered_set<Triangle> triangles = delaunay(points, indicesRight, partition, bounds2, edgesActive2, depth + 1);
        profiler::startBranch(depth);
        profiler::startSection(depth, profiler::MERGE);
        output.insert(triangles.begin(), triangles.end());
        profiler::stopSection(depth, profiler::MERGE);
        profiler::stopBranch(depth);
    }

    return output;
}

int main(){
    vector<Point> points;

    // Load points from file
    cout << "Loading points from file..." << endl;
    ifstream filePoints("../points.dat");
    if(!filePoints.is_open()) {
        cerr << "Failed to open points file!" << endl;
        return 1;
    }
    string filePointsLine;
    int indexLine = 0;
    while(getline(filePoints, filePointsLine)){
        istringstream iss(filePointsLine);
        Point point = {0.0f, 0.0f};
        iss >> point.x >> point.y;
        points.push_back(point);
        indexLine++;
    }
    filePoints.close();

    // Pre-process input data
    cout << "Partitioning input points..." << endl;
    vector<int> indices;
    Partition partition;
    for(int i = 0; i < points.size(); i++){
        indices.push_back(i);
        partition.insert(points[i], i);
    }

    // Run delaunay triangulation
    cout << "Starting triangulation..." << endl;
    profiler::startProgram();
    unordered_set<Triangle> connections = delaunay(points, indices, partition, Bounds(0.0f, 1.0f, 0.0f, 1.0f), {}, 0);
    profiler::stopProgram();

    // Write connections to file
    cout << "Writing connections to file..." << endl;
    ofstream fileConnections("../connections.dat");
    if(!fileConnections.is_open()){
        cerr << "Failed to open connections file!" << endl;
        return 1;
    }
    for(Triangle triangle : connections){
        fileConnections << triangle.i1 << " " << triangle.i2 << " " << triangle.i3 << " " << endl;
    }
    fileConnections.close();

    return 0;
}
