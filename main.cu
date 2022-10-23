#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "primitive.cuh"
#include "dewall.cuh"
#include "blelloch.cuh"
#include "profiler.cuh"

using namespace std;

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

    //profiler::startProgram(profiler::sectionsDeWall);
    //unordered_set<Triangle> connections = dewall::triangulate(points, indices, partition, Bounds(0.0f, 1.0f, 0.0f, 1.0f), {}, 0);

    profiler::startProgram(profiler::sectionsBlelloch);
    unordered_set<Triangle> connections = blelloch::triangulate(points, indices);

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
