#pragma once

#include <iostream>
#include <chrono>

using namespace std;

namespace profiler_circles{

    void startProgram(const string& fileNameArg);
    void stopProgram();

    void initializeBatch(int size);
    void startBatch(int size);
    void stopBatch(int size);

}