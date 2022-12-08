#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <functional>

#include "src/primitive.cuh"
#include "src/circles_noop.cuh"
#include "src/circles_serial.cuh"
#include "src/circles_parallel.cuh"
#include "src/mesh_dewall.cuh"
#include "src/mesh_blelloch.cuh"
#include "src/profiler_circles.cuh"
#include "src/profiler_circles.cu"
#include "src/profiler_mesh.cuh"
#include "src/profiler_mesh.cu"

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

void saveTriangles(const unordered_set<Triangle>& triangles){
    ofstream fileConnections("../connections.dat");
    if(!fileConnections.is_open()) throw runtime_error("Failed to open connections file!");
    for(Triangle triangle : triangles){
        fileConnections << triangle.i1 << " " << triangle.i2 << " " << triangle.i3 << " " << endl;
    }
    fileConnections.close();
}

int runCirclesTest(const string& fileName,
                   const function<void(const vector<Point>&, const Triangle&, const int&, const int&)>& funcTestSpeed,
                   const function<vector<bool>(const vector<Point>&, const Triangle&, const int&, const int&)>& funcTestAccuracy){
    // Seed the random so we get the same sequence of triangles across multiple tests
    srand(69420);

    // Load points from file
    cout << "Loading points from file..." << endl;
    vector<Point> points = loadPoints();

    // Pre-process input data
    cout << "Indexing input points..." << endl;
    vector<int> indices;
    for(int i = 0; i < points.size(); i++){
        indices.push_back(i);
    }

    // Create triangles
    cout << "Selecting triangles..." << endl;
    vector<Triangle> triangles;
    for(int i = 0; i < 100; i++){
        Triangle triangle{};
        triangle.i1 = rand() % points.size();
        do{
            triangle.i2 = rand() % points.size();
        }while(triangle.i2 == triangle.i1);
        do{
            triangle.i3 = rand() % points.size();
        }while(triangle.i3 == triangle.i1 || triangle.i3 == triangle.i2);
        triangles.push_back(makeClockwise(points, triangle));
    }
    vector<int> offsets;
    for(int i = 0; i < 100; i++){
        offsets.push_back(rand() % points.size());
    }

    cout << "Running circumcircle tests..." << endl;
    profiler_circles::startProgram(fileName);
    for(int batchPower = 0; batchPower <= 3; batchPower++){
        int batchSizeBase = (int)pow(10, batchPower);
        for(int batchSize : {batchSizeBase, 2 * batchSizeBase, 5 * batchSizeBase}){
        //for(int batchSize = batchSizeBase; batchSize < batchSizeBase * 10; batchSize += batchSizeBase){

            cout << "Batch size: " << batchSize << endl;

            // Run circumcircle speed test

            profiler_circles::startBatch(batchSize);
            for(int t = 0; t < triangles.size(); t++){
                funcTestSpeed(points, triangles[t], offsets[t], batchSize);
            }
            profiler_circles::stopBatch(batchSize);

            // Run circumcircle accuracy test
            int errors = 0;
            for(int t = 0; t < triangles.size(); t++){
                vector<bool> outputTest = funcTestAccuracy(points, triangles[t], offsets[t], batchSize);
                vector<bool> outputReference = circles_serial::testPureAccuracy(points, triangles[t], offsets[t], batchSize);
                for(int i = 0; i < outputReference.size(); i++){
                    if(outputTest[i] != outputReference[i]) errors++;
                }
            }
            if(errors > 0) cerr << "Failed " << errors << " accuracy tests!" << endl;
        }
    }
    profiler_circles::stopProgram();

    return 0;
}

int runMeshGeneration(const string& fileName, const vector<profiler_mesh::Section>& profilerSections,
                      const function<vector<Triangle>(const vector<Point>&, const vector<int>&)>& func){
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
    profiler_mesh::startProgram(fileName, profilerSections);
    vector<Triangle> connections = func(points, indices);
    profiler_mesh::stopProgram();

    // Post-process output data
    cout << "Verifying output triangles..." << endl;
    unordered_set<Triangle> triangles;
    int duplicates = 0;
    for(Triangle triangle : connections){
        Triangle triangleOutput = makeSequential(points, {triangle.i1, triangle.i2, triangle.i3});
        if(triangles.count(triangleOutput)) duplicates++;
        triangles.insert(triangleOutput);
    }
    if(duplicates > 0) cerr << "Generated " << duplicates << " duplicate triangle(s)!" << endl;

    // Write triangles to file
    cout << "Writing triangles to file..." << endl;
    saveTriangles(triangles);

    return 0;
}

int main(){
    runCirclesTest("noop", circles_noop::testSpeed, circles_noop::testAccuracy);
    //runCirclesTest("serial_pure", circles_serial::testPureSpeed, circles_serial::testPureAccuracy);
    //runCirclesTest("serial_regions", circles_serial::testRegionsSpeed, circles_serial::testRegionsAccuracy);
    //runCirclesTest("serial_dist", circles_serial::testDistanceSpeed, circles_serial::testDistanceAccuracy);
    runCirclesTest("parallel", circles_parallel::testSpeed, circles_parallel::testAccuracy);
    //return runMeshGeneration("dewall", profiler_mesh::sectionsMeshDeWall, mesh_dewall::triangulate);
    //return runMeshGeneration("blelloch", profiler_mesh::sectionsMeshBlelloch, mesh_blelloch::triangulate);
}
