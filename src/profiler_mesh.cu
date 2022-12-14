#include <iostream>
#include <fstream>
#include <chrono>
#include <unordered_map>

#include "profiler_mesh.cuh"

using namespace std;

#define Time chrono::high_resolution_clock::time_point

namespace profiler_mesh{

    inline auto now(){
        return chrono::high_resolution_clock::now();
    }

    inline auto elapsedNano(Time timeStart, Time timeEnd){
        return chrono::duration_cast<chrono::nanoseconds>(timeEnd - timeStart).count();
    }

    string fileName;
    vector<Section> sectionsActive;
    Time timeProgramStart, timeLocalBranch, timeLocalSection;
    vector<unordered_map<Section, unsigned long long>> times;

    void startProgram(const string& fileNameArg, const vector<Section>& sectionsActiveArg){
        fileName = fileNameArg;
        sectionsActive = sectionsActiveArg;
        timeProgramStart = now();
    }

    void stopProgram(){
        auto elapsedNanoTotal = elapsedNano(timeProgramStart, now());

        for(auto& time : times){
            for(Section section : sectionsActive){
                if(section != OTHER && time.count(section)){
                    time[OTHER] -= time[section];
                }
            }
        }

        cout << "Writing profile to file..." << endl;
        ofstream fileProfile("../profile_" + fileName + "_mesh.csv");
        if(!fileProfile.is_open()){
            cerr << "Failed to open profile file!" << endl;
        }

        fileProfile << elapsedNanoTotal;
        for(Section section : sectionsActive){
            fileProfile << "," << section;
        }
        fileProfile << "\n";

        for(int i = 0; i < times.size(); i++){
            fileProfile << i;
            for(Section section : sectionsActive){
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

    void startSection(int depth, const Section section){
        if(times.size() <= depth){
            unordered_map<Section, unsigned long long> timeMap;
            times.push_back(timeMap);
        }
        timeLocalSection = now();
    }

    void stopSection(int depth, const Section section){
        times[depth][section] += elapsedNano(timeLocalSection, now());
    }

}