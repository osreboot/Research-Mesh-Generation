#pragma comment(lib, "cublas.lib")

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <functional>

#include "src/primitive.cuh"
#include "src/circles.cu"
#include "src/circles_serial_noop.cu"
#include "src/circles_serial_pure.cu"
#include "src/circles_serial_disjoint.cu"
#include "src/circles_serial_regions.cu"
#include "src/circles_serial_distance.cu"
#include "src/circles_serial_shotgun.cu"
#include "src/circles_kernel_noop.cu"
#include "src/circles_kernel_pure.cu"
#include "src/circles_kernel_distance.cu"
#include "src/circles_kernel_shotgun.cu"
#include "src/circles_tensor_distance.cu"
#include "src/circles_tensor_disjoint.cu"
#include "src/circles_tensor_shotgun.cu"
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

int runCirclesTest(const shared_ptr<Circles>& circles){
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

    CirclesSerialPure circlesRef;

    cout << "Running circumcircle tests (" << circles->getFileName() << ")..." << endl;
    profiler_circles::startProgram(circles->getFileName());
    //for(int batchSize = 16; batchSize < 3000000; batchSize *= 2){
    for(int batchPower = 0; batchPower <= 5; batchPower++){
        int batchSizeBase = (int)pow(10, batchPower);
        for(int batchSize : {batchSizeBase, 2 * batchSizeBase, 5 * batchSizeBase}){
        //for(int batchSize = batchSizeBase; batchSize < batchSizeBase * 10; batchSize += batchSizeBase){

            cout << "Batch size: " << batchSize << endl;
            profiler_circles::initializeBatch(batchSize);

            int errors = 0;
            for(int t = 0; t < triangles.size(); t++){

                auto *batchPoints = new Point[batchSize];
                for(int i = 0; i < batchSize; i++){
                    batchPoints[i] = points[(offsets[t] + i) % points.size()];
                }
                circles->initialize(batchPoints, batchSize);
                circlesRef.initialize(batchPoints, batchSize);

                // Run circumcircle test
                auto *batchOutput = new bool[batchSize];
                profiler_circles::startBatch(batchSize);
                circles->run(batchOutput, points[triangles[t].i1], points[triangles[t].i2], points[triangles[t].i3]);
                profiler_circles::stopBatch(batchSize);

                // Run reference
                auto *batchOutputRef = new bool[batchSize];
                circlesRef.run(batchOutputRef, points[triangles[t].i1], points[triangles[t].i2], points[triangles[t].i3]);

                for(int i = 0; i < batchSize; i++){
                    if(batchOutput[i] != batchOutputRef[i]) errors++;
                }

                circles->cleanup();
                circlesRef.cleanup();

                delete[] batchPoints;
                delete[] batchOutput;
                delete[] batchOutputRef;
            }

            if(errors > 0) cout << "ERROR: Failed " << errors << " accuracy tests! (" << ((float)errors / (float)(triangles.size() * batchSize)) * 100.0f << "%)" << endl;
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
    if(duplicates > 0) cout << "ERROR: Generated " << duplicates << " duplicate triangle(s)!" << endl;

    // Write triangles to file
    cout << "Writing triangles to file..." << endl;
    saveTriangles(triangles);

    return 0;
}

int main(){
    //runCirclesTest(make_shared<CirclesNoop>());
    //runCirclesTest(make_shared<CirclesSerialPure>());
    //runCirclesTest(make_shared<CirclesSerialDisjoint>());
    //runCirclesTest(make_shared<CirclesSerialRegions>());
    //runCirclesTest(make_shared<CirclesSerialDistance>());
    //runCirclesTest(make_shared<CirclesSerialShotgun>());
    //runCirclesTest(make_shared<CirclesKernelNoop>());
    //runCirclesTest(make_shared<CirclesKernelPure>());
    //runCirclesTest(make_shared<CirclesKernelDistance>());
    //runCirclesTest(make_shared<CirclesKernelShotgun>());
    //runCirclesTest(make_shared<CirclesTensorDistance>());
    //runCirclesTest(make_shared<CirclesTensorDisjoint>());
    runCirclesTest(make_shared<CirclesTensorShotgun>());

    //return runMeshGeneration("dewall", profiler_mesh::sectionsMeshDeWall, mesh_dewall::triangulate);
    //return runMeshGeneration("blelloch", profiler_mesh::sectionsMeshBlelloch, mesh_blelloch::triangulate);
}
