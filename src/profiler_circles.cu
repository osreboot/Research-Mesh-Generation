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
    Time timeLocal;
    map<int, unsigned long long> timesLoad, timesRun, timesSave;

    void startProgram(const string& fileNameArg){
        fileName = fileNameArg;
    }

    void stopProgram(){
        cout << "Writing profile to file..." << endl;
        ofstream fileProfile("../profile_" + fileName + "_circles.csv");
        if(!fileProfile.is_open()) cerr << "Failed to open profile file!" << endl;

        for(const auto& pair : timesLoad) fileProfile << "," << pair.first;
        fileProfile << "\n" << fileName << "[load]";
        for(const auto& pair : timesLoad) fileProfile << "," << pair.second;
        fileProfile << "\n" << fileName << "[run]";
        for(const auto& pair : timesRun) fileProfile << "," << pair.second;
        fileProfile << "\n" << fileName << "[save]";
        for(const auto& pair : timesSave) fileProfile << "," << pair.second;
        fileProfile.close();
    }

    void initializeSections(int size){
        timesLoad[size] = 0;
        timesRun[size] = 0;
        timesSave[size] = 0;
    }

    void startSection(){
        timeLocal = now();
    }

    void stopLoad(int size){
        timesLoad[size] += elapsedNano(timeLocal, now());
    }

    void stopRun(int size){
        timesRun[size] += elapsedNano(timeLocal, now());
    }

    void stopSave(int size){
        timesSave[size] += elapsedNano(timeLocal, now());
    }

}