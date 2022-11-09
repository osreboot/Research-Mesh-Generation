#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <functional>

#include "src/primitive.cuh"
#include "src/circles_linear.cuh"
#include "src/mesh_dewall.cuh"
#include "src/mesh_blelloch.cuh"
#include "src/profiler.cuh"
#include "src/profiler.cu"

using namespace std;

vector<Point> loadPoints(){
    vector<Point> points;

    ifstream filePoints("../points.dat");
    if(!filePoints.is_open()) throw runtime_error("Failed to open points file!");
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

    return points;
}

void saveConnections(const unordered_set<Triangle>& connections){
    ofstream fileConnections("../connections.dat");
    if(!fileConnections.is_open()) throw runtime_error("Failed to open connections file!");
    for(Triangle triangle : connections){
        fileConnections << triangle.i1 << " " << triangle.i2 << " " << triangle.i3 << " " << endl;
    }
    fileConnections.close();
}

int runCirclesTest(const vector<profiler::Section>& profilerSections,
                   const function<void(const vector<Point>&, const int&)>& func){
    // Load points from file
    cout << "Loading points from file..." << endl;
    vector<Point> points = loadPoints();

    // Pre-process input data
    cout << "Indexing input points..." << endl;
    vector<int> indices;
    for(int i = 0; i < points.size(); i++){
        indices.push_back(i);
    }

    // Run circumcircle test
    cout << "Starting circumcircle test..." << endl;
    for(int i = 0; i < (points.size() / 10); i += 3){
        func(points, i);
    }

    return 0;
}

int runMeshGeneration(const vector<profiler::Section>& profilerSections,
                      const function<unordered_set<Triangle>(const vector<Point>&, const vector<int>&)>& func){
    // Load points from file
    cout << "Loading points from file..." << endl;
    vector<Point> points = loadPoints();

    // Pre-process input data
    cout << "Indexing input points..." << endl;
    vector<int> indices;
    for(int i = 0; i < points.size(); i++){
        indices.push_back(i);
    }

    // Run delaunay triangulation
    cout << "Starting triangulation..." << endl;
    profiler::startProgram(profilerSections);
    unordered_set<Triangle> connections = func(points, indices);
    profiler::stopProgram();

    // Write connections to file
    cout << "Writing connections to file..." << endl;
    saveConnections(connections);

    return 0;
}

int main(){
    //return runCirclesTest(profiler::sectionsMeshBlelloch, circles_linear::circles);
    return runMeshGeneration(profiler::sectionsMeshDeWall, mesh_dewall::triangulate);
    //return runMeshGeneration(profiler::sectionsMeshBlelloch, mesh_blelloch::triangulate);
}
