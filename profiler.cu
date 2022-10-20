#include <iostream>
#include <fstream>
#include <chrono>
#include <unordered_map>

#include "profiler.cuh"

using namespace std;

#define Time chrono::high_resolution_clock::time_point

inline auto now(){
    return chrono::high_resolution_clock::now();
}

inline auto elapsedNano(Time timeStart, Time timeEnd){
    return chrono::duration_cast<chrono::nanoseconds>(timeEnd - timeStart).count();
}

namespace profiler{

    Time timeProgramStart, timeLocalBranch, timeLocalSection;
    vector<unordered_map<Section, unsigned long long>> times;

    void startProgram(){
        timeProgramStart = now();
    }

    void stopProgram(){
        auto elapsedNanoTotal = elapsedNano(timeProgramStart, now());

        for(int i = 0; i < times.size(); i++){
            for(Section section : sections){
                if(section != OTHER && times[i].count(section)){
                    times[i][OTHER] -= times[i][section];
                }
            }
        }

        cout << "Writing profile to file..." << endl;
        ofstream fileProfile("../profile.csv");
        if(!fileProfile.is_open()){
            cerr << "Failed to open profile file!" << endl;
        }

        fileProfile << elapsedNanoTotal;
        for(Section section : sections){
            fileProfile << "," << section;
        }
        fileProfile << "\n";

        for(int i = 0; i < times.size(); i++){
            fileProfile << i;
            for(Section section : sections){
                if(times[i].count(section)){
                    fileProfile << "," << times[i][section];
                }else fileProfile << "," << 0;
            }
            fileProfile << "\n";
        }

        fileProfile.close();
    }

    void startBranch(int depth){
        if(times.size() <= depth){
            unordered_map<Section, unsigned long long> timeMap;
            times.push_back(timeMap);
        }
        timeLocalBranch = now();
    }

    void stopBranch(int depth){
        times[depth][OTHER] += elapsedNano(timeLocalBranch, now());
    }

    void startSection(int depth, Section section){
        if(times.size() <= depth){
            unordered_map<Section, unsigned long long> timeMap;
            times.push_back(timeMap);
        }
        timeLocalSection = now();
    }

    void stopSection(int depth, Section section){
        times[depth][section] += elapsedNano(timeLocalSection, now());
    }

}