#pragma once

#include <iostream>
#include <chrono>

using namespace std;

namespace profiler_circles{

    void startProgram(const string& fileNameArg);
    void stopProgram();

    void initializeSections(int size);
    void startSection();
    void stopLoad(int size);
    void stopRun(int size);
    void stopSave(int size);

}