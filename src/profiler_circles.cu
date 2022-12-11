#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

#include "profiler_circles.cuh"

using namespace std;

#define Time chrono::high_resolution_clock::time_point

namespace profiler_circles{

    inline auto now(){
        return chrono::high_resolution_clock::now();
    }

    inline auto elapsedNano(Time timeStart, Time timeEnd){
        return chrono::duration_cast<chrono::nanoseconds>(timeEnd - timeStart).count();
    }

    string fileName;
    Time timeLocalBatch;
    map<int, unsigned long long> times;

    void startProgram(const string& fileNameArg){
        fileName = fileNameArg;
    }

    void stopProgram(){
        cout << "Writing profile to file..." << endl;
        ofstream fileProfile("../profile_" + fileName + "_circles.csv");
        if(!fileProfile.is_open()){
            cerr << "Failed to open profile file!" << endl;
        }

        for(const auto& pair : times){
            fileProfile << "," << pair.first;
        }
        fileProfile << "\n" << fileName;
        for(const auto& pair : times){
            fileProfile << "," << pair.second;
        }
        fileProfile.close();
    }

    void initializeBatch(int size){
        times[size] = 0;
    }

    void startBatch(int size){
        timeLocalBatch = now();
    }

    void stopBatch(int size){
        times[size] += elapsedNano(timeLocalBatch, now());
    }

}