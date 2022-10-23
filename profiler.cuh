#pragma once

#include <iostream>
#include <chrono>

using namespace std;

namespace profiler{

    enum Section{
        WALL,
        INIT,
        L_FIRST,
        L_CIRCLE,
        PREP,
        REPLACE,
        SAVE,
        REDACT,
        CHAIN,
        MERGE,
        POST,
        OTHER
    };

    const vector<Section> sectionsDeWall = {WALL, INIT, L_FIRST, L_CIRCLE, SAVE, CHAIN, MERGE, OTHER};
    const vector<Section> sectionsBlelloch = {INIT, PREP, REPLACE, SAVE, REDACT, POST, OTHER};

    inline ostream& operator<<(ostream& ostr, Section section){
        switch(section){
            case WALL: return ostr << "WALL";
            case INIT: return ostr << "INIT";
            case L_FIRST: return ostr << "L_FIRST";
            case L_CIRCLE: return ostr << "L_CIRCLE";
            case PREP: return ostr << "PREP";
            case REPLACE: return ostr << "REPLACE";
            case SAVE: return ostr << "SAVE";
            case REDACT: return ostr << "REDACT";
            case CHAIN: return ostr << "CHAIN";
            case MERGE: return ostr << "MERGE";
            case POST: return ostr << "POST";
            case OTHER: return ostr << "OTHER";
            default: return ostr << "UNKNOWN";
        }
    }

    void startProgram(const vector<Section>& sectionsActiveArg);
    void stopProgram();

    void startBranch(int depth);
    void stopBranch(int depth);

    void startSection(int depth, Section section);
    void stopSection(int depth, Section section);

}