#pragma once

#include <iostream>
#include <chrono>

using namespace std;

namespace profiler_mesh{

    enum Section{
        CREATE_WALL,
        INIT,
        INIT_DIVIDE_POINTS,
        INIT_FIRST_EDGE,
        INIT_LISTS,
        DIVIDE_EDGES,
        L_FIRST,
        L_CIRCLE,
        LOCATE,
        DIVIDE_POINTS,
        FIRST_EDGE,
        CIRCLES_LOAD,
        CIRCLES_RUN,
        CIRCLES_SAVE,
        CIRCLES_CLEANUP,
        PREP,
        REPLACE,
        SAVE_TRIANGLES,
        UPDATE_EDGES,
        REDACT,
        CHAIN,
        MERGE,
        POST,
        OTHER
    };

    const vector<Section> sectionsMeshDewall = {CREATE_WALL, DIVIDE_POINTS, FIRST_EDGE, INIT_LISTS, DIVIDE_EDGES,
                                                CIRCLES_LOAD, CIRCLES_RUN, CIRCLES_SAVE, CIRCLES_CLEANUP,
                                                SAVE_TRIANGLES, UPDATE_EDGES, OTHER};
    const vector<Section> sectionsMeshDewallOld = {CREATE_WALL, INIT_DIVIDE_POINTS, INIT_FIRST_EDGE, INIT_LISTS,
                                                   DIVIDE_EDGES, L_FIRST, L_CIRCLE, SAVE_TRIANGLES, CHAIN, MERGE, OTHER};
    const vector<Section> sectionsMeshBlellochOld = {INIT, PREP, REPLACE, SAVE_TRIANGLES, REDACT, POST, OTHER};

    inline ostream& operator<<(ostream& ostr, Section section){
        switch(section){
            case CREATE_WALL: return ostr << "CREATE_WALL";
            case INIT: return ostr << "INIT";
            case INIT_DIVIDE_POINTS: return ostr << "INIT_DIVIDE_POINTS";
            case INIT_FIRST_EDGE: return ostr << "INIT_FIRST_EDGE";
            case INIT_LISTS: return ostr << "INIT_LISTS";
            case DIVIDE_EDGES: return ostr << "DIVIDE_EDGES";
            case L_FIRST: return ostr << "L_FIRST";
            case L_CIRCLE: return ostr << "L_CIRCLE";
            case LOCATE: return ostr << "LOCATE";
            case DIVIDE_POINTS: return ostr << "DIVIDE_POINTS";
            case FIRST_EDGE: return ostr << "FIRST_EDGE";
            case CIRCLES_LOAD: return ostr << "CIRCLES_LOAD";
            case CIRCLES_RUN: return ostr << "CIRCLES_RUN";
            case CIRCLES_SAVE: return ostr << "CIRCLES_SAVE";
            case CIRCLES_CLEANUP: return ostr << "CIRCLES_CLEANUP";
            case PREP: return ostr << "PREP";
            case REPLACE: return ostr << "REPLACE";
            case SAVE_TRIANGLES: return ostr << "SAVE_TRIANGLES";
            case UPDATE_EDGES: return ostr << "UPDATE_EDGES";
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