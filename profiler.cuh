#pragma once

#include <iostream>
#include <chrono>

namespace profiler{

    enum Section{
        WALL,
        INIT,
        L_FIRST,
        L_CIRCLE,
        SAVE,
        CHAIN,
        MERGE,
        OTHER
    };

    const Section sections[] = {WALL, INIT, L_FIRST, L_CIRCLE, SAVE, CHAIN, MERGE, OTHER};

    inline std::ostream& operator<<(std::ostream& ostr, Section section){
        switch(section){
            case WALL: return ostr << "WALL";
            case INIT: return ostr << "INIT";
            case L_FIRST: return ostr << "L_FIRST";
            case L_CIRCLE: return ostr << "L_CIRCLE";
            case SAVE: return ostr << "SAVE";
            case CHAIN: return ostr << "CHAIN";
            case MERGE: return ostr << "MERGE";
            case OTHER: return ostr << "OTHER";
            default: return ostr << "UNKNOWN";
        }
    }

    void startProgram();
    void stopProgram();

    void startBranch(int depth);
    void stopBranch(int depth);

    void startSection(int depth, Section section);
    void stopSection(int depth, Section section);

}