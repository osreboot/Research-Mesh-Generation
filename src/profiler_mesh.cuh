#pragma once

#include <iostream>
#include <chrono>

using namespace std;

namespace profiler_mesh{

    enum Section{
        WALL,
        INIT,
        INIT_DIVIDE_POINTS,
        INIT_FIRST_EDGE,
        INIT_LISTS,
        INIT_INHERIT_EDGES,
        L_FIRST,
        L_CIRCLE,
        LOCATE,
        DIVIDE_POINTS,
        FIRST_EDGE,
        PREP,
        REPLACE,
        SAVE,
        REDACT,
        CHAIN,
        MERGE,
        POST,
        OTHER
    };

    const vector<Section> sectionsMeshDewall = {WALL, DIVIDE_POINTS, FIRST_EDGE, INIT_LISTS, INIT_INHERIT_EDGES, PREP, LOCATE, SAVE, CHAIN, OTHER};
    const vector<Section> sectionsMeshDeWallOld = {WALL, INIT_DIVIDE_POINTS, INIT_FIRST_EDGE, INIT_LISTS, INIT_INHERIT_EDGES, L_FIRST, L_CIRCLE, SAVE, CHAIN, MERGE, OTHER};
    const vector<Section> sectionsMeshBlellochOld = {INIT, PREP, REPLACE, SAVE, REDACT, POST, OTHER};

    inline ostream& operator<<(ostream& ostr, Section section){
        switch(section){
            case WALL: return ostr << "WALL";
            case INIT: return ostr << "INIT";
            case INIT_DIVIDE_POINTS: return ostr << "INIT_DIVIDE_POINTS";
            case INIT_FIRST_EDGE: return ostr << "INIT_FIRST_EDGE";
            case INIT_LISTS: return ostr << "INIT_LISTS";
            case INIT_INHERIT_EDGES: return ostr << "INIT_INHERIT_EDGES";
            case L_FIRST: return ostr << "L_FIRST";
            case L_CIRCLE: return ostr << "L_CIRCLE";
            case LOCATE: return ostr << "LOCATE";
            case DIVIDE_POINTS: return ostr << "DIVIDE_POINTS";
            case FIRST_EDGE: return ostr << "FIRST_EDGE";
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

    void startProgram(const string& fileNameArg, const vector<Section>& sectionsActiveArg);
    void stopProgram();

    void startBranch(int depth);
    void stopBranch(int depth);

    void startSection(int depth, Section section);
    void stopSection(int depth, Section section);

}