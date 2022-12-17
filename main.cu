#pragma comment(lib, "cublas.lib")

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <functional>
#include <cassert>

#include "src/util.cuh"
#include "src/primitive.cuh"
#include "src/circles.cu"
#include "src/circles_serial_noop.cu"
#include "src/circles_serial_pure.cu"
#include "src/circles_serial_regions.cu"
#include "src/circles_serial_distance.cu"
#include "src/circles_serial_lightweight.cu"
#include "src/circles_kernel_noop.cu"
#include "src/circles_kernel_pure.cu"
#include "src/circles_kernel_distance.cu"
#include "src/circles_kernel_lightweight.cu"
#include "src/circles_ntensor_lightweight.cu"
#include "src/circles_ntensor_twostep.cu"
#include "src/circles_ptensor_lightweight.cu"
#include "src/circles_ptensor_twostep.cu"
#include "src/circles_tensor_lightweight.cu"
#include "src/circles_tensor_twostep.cu"
#include "src/mesh.cu"
#include "src/mesh_dewall.cu"
#include "src/mesh_dewall_old.cuh"
#include "src/mesh_blelloch_old.cuh"
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

    cudaDeviceReset();

    cout << "Running circumcircle tests (" << circles->getFileName() << ")..." << endl;
    profiler_circles::startProgram(circles->getFileName());
    //for(int batchSize = 16; batchSize < 3000000; batchSize *= 2){
    for(int batchPower = 0; batchPower <= 6; batchPower++){
        int batchSizeBase = (int)pow(10, batchPower);
        //for(int batchSize : {batchSizeBase, 2 * batchSizeBase, 5 * batchSizeBase}){
        //for(int batchSize : {batchSizeBase, 2 * batchSizeBase, 4 * batchSizeBase, 8 * batchSizeBase}){
        //for(int batchSize = batchSizeBase; batchSize < batchSizeBase * 10; batchSize += batchSizeBase){
        for(double d = 0.0; d < 1.0 - 0.1; d += 0.1){
            int batchSize = (int)((double)batchSizeBase * (pow(10, d)));

            cout << "Batch size: " << batchSize << endl;
            profiler_circles::initializeSections(batchSize);

            int errors = 0;
            for(int t = 0; t < triangles.size(); t++){

                auto *px = new double[batchSize];
                auto *py = new double[batchSize];
                for(int i = 0; i < batchSize; i++){
                    px[i] = points[(offsets[t] + i) % points.size()].x;
                    py[i] = points[(offsets[t] + i) % points.size()].y;
                }

                //cudaDeviceReset();

                // Load test data
                profiler_circles::startSection();
                circles->load(px, py, batchSize);
                cudaDeviceSynchronize();
                profiler_circles::stopLoad(batchSize);

                // Run test
                auto *batchOutput = new bool[batchSize];
                profiler_circles::startSection();
                circles->run(batchOutput, points[triangles[t].i1], points[triangles[t].i2], points[triangles[t].i3]);
                cudaDeviceSynchronize();
                profiler_circles::stopRun(batchSize);

                // Save test data
                profiler_circles::startSection();
                circles->save(batchOutput);
                cudaDeviceSynchronize();
                profiler_circles::stopSave(batchSize);

                // Run reference
                auto *batchOutputRef = new bool[batchSize];
                circlesRef.load(px, py, batchSize);
                circlesRef.run(batchOutputRef, points[triangles[t].i1], points[triangles[t].i2], points[triangles[t].i3]);
                circlesRef.save(batchOutputRef);
                cudaDeviceSynchronize();

                for(int i = 0; i < batchSize; i++){
                    if(batchOutput[i] != batchOutputRef[i]) errors++;
                }

                circles->cleanup();
                circlesRef.cleanup();

                delete[] px;
                delete[] py;

                delete[] batchOutput;
                delete[] batchOutputRef;
            }

            if(errors > 0) cout << "ERROR: Failed " << errors << " accuracy tests! (" << ((float)errors / (float)(triangles.size() * batchSize)) * 100.0f << "%)" << endl;
        }
    }
    profiler_circles::stopProgram();

    return 0;
}

int runMeshGeneration(const shared_ptr<Mesh>& mesh){
        //const string& fileName, const vector<profiler_mesh::Section>& profilerSections,
        //              const function<vector<Triangle>(const vector<Point>&, const vector<int>&)>& func){
    // Load points from file
    cout << "Loading points from file..." << endl;
    vector<Point> points = loadPoints();
    auto *px = new double[points.size()];
    auto *py = new double[points.size()];
    for(int i = 0; i < points.size(); i++){
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    auto *connections = new Triangle[3 * points.size() - 4]; // Extra padding so we don't get a memory access error
                                                             // (below assert catches that)
    int connectionsSize = 0;

    // Run delaunay triangulation
    cout << "Starting triangulation (" << mesh->getFileName() << ")..." << endl;
    profiler_mesh::startProgram(mesh->getFileName(), mesh->getProfilerSections());
    mesh->run(connections, connectionsSize, px, py, points.size());
    profiler_mesh::stopProgram();

    // Post-process output data
    cout << "Verifying output triangles..." << endl;
    assert(connectionsSize <= 2 * points.size() - 4);
    assert(connectionsSize > 0);
    unordered_set<Triangle> triangles;
    triangles.reserve(connectionsSize);
    int duplicates = 0;
    for(int i = 0; i < connectionsSize; i++){
        Triangle triangleOutput = makeSequential(points, {connections[i].i1, connections[i].i2, connections[i].i3});
        if(triangles.count(triangleOutput)) duplicates++;
        triangles.insert(triangleOutput);
    }
    if(duplicates > 0) cout << "ERROR: Generated " << duplicates << " duplicate triangle(s)!" << endl;

    // Write triangles to file
    cout << "Writing triangles to file..." << endl;
    saveTriangles(triangles);

    delete[] connections;

    delete[] px;
    delete[] py;

    return 0;
}

int main(){
    //runCirclesTest(make_shared<CirclesSerialNoop>());
    //runCirclesTest(make_shared<CirclesSerialPure>());
    //runCirclesTest(make_shared<CirclesSerialRegions>());
    //runCirclesTest(make_shared<CirclesSerialDistance>());
    //runCirclesTest(make_shared<CirclesSerialLightweight>());
    //runCirclesTest(make_shared<CirclesKernelNoop>());
    //runCirclesTest(make_shared<CirclesKernelPure>());
    //runCirclesTest(make_shared<CirclesKernelDistance>());
    //runCirclesTest(make_shared<CirclesKernelLightweight>());
    //runCirclesTest(make_shared<CirclesNTensorLightweight>());
    //runCirclesTest(make_shared<CirclesNTensorTwostep>());
    runCirclesTest(make_shared<CirclesPTensorLightweight>());
    runCirclesTest(make_shared<CirclesPTensorTwostep>());
    runCirclesTest(make_shared<CirclesTensorLightweight>());
    runCirclesTest(make_shared<CirclesTensorTwostep>());

    //return runMeshGeneration(make_shared<MeshDewall>());
    //return runMeshGeneration("dewall", profiler_mesh::sectionsMeshDewallOld, mesh_dewall_old::triangulate);
    //return runMeshGeneration("blelloch", profiler_mesh::sectionsMeshBlellochOld, mesh_blelloch_old::triangulate);
}
