#pragma once

#include <unordered_set>
#include <unordered_map>

#include "primitive.cuh"
#include "math.cuh"
#include "profiler.cuh"

using namespace std;

namespace mesh_blelloch{

    class TriangleB{

    public:
        int i1, i2, i3;
        unordered_set<int> pointsEnc;

        TriangleB(int i1Arg, int i2Arg, int i3Arg){
            i1 = i1Arg;
            i2 = i2Arg;
            i3 = i3Arg;

            pointsEnc = {};
        }

        unordered_set<Edge> getEdges() const {
            return {{i1, i2}, {i2, i3}, {i3, i1}};
        }

    };

    vector<shared_ptr<TriangleB>> replace(const vector<Point>& points,
                                          const unordered_map<Edge, shared_ptr<TriangleB>>& edgeLinks,
                                          const unordered_set<Edge>& edgesReplacing,
                                          const TriangleB& triangle, int point){
        vector<shared_ptr<TriangleB>> output;
        for(Edge edge : triangle.getEdges()){
            if(!edgesReplacing.count(edge.reverse())){
                shared_ptr<TriangleB> triangleNew(new TriangleB(edge.i1, edge.i2, point));
                for(int indexEnc : triangle.pointsEnc){
                    if(indexEnc != point && isInCircle(points[triangleNew->i1], points[triangleNew->i2], points[triangleNew->i3], points[indexEnc])){
                        triangleNew->pointsEnc.insert(indexEnc);
                    }
                }
                if(edgeLinks.count(edge.reverse())){
                    for(int indexEnc : (*edgeLinks.find(edge.reverse())).second->pointsEnc){
                        if(indexEnc != point && !triangleNew->pointsEnc.count(indexEnc) &&
                           isInCircle(points[triangleNew->i1], points[triangleNew->i2], points[triangleNew->i3], points[indexEnc])){
                            triangleNew->pointsEnc.insert(indexEnc);
                        }
                    }
                }
                output.push_back(triangleNew);
            }
        }
        return output;
    }

    vector<Triangle> triangulate(const vector<Point>& points, const vector<int>& indices){
        unordered_set<shared_ptr<TriangleB>> trianglesOutput;
        unordered_map<Edge, shared_ptr<TriangleB>> edgeLinks;
        unordered_map<int, unordered_set<shared_ptr<TriangleB>>> pointsEnc;

        // Create super triangle
        profiler::startBranch(0);
        profiler::startSection(0, profiler::INIT);
        vector<Point> pointsS = points;
        pointsS.push_back({-10000000000.0f, -10000000000.0f});
        pointsS.push_back({-10000000000.0f, 20000000000.0f});
        pointsS.push_back({20000000000.0f, -10000000000.0f});
        shared_ptr<TriangleB> triangleSuper(new TriangleB(indices.size(), indices.size() + 1, indices.size() + 2));
        for(int i : indices){
            triangleSuper->pointsEnc.insert(i);
            pointsEnc.insert({i, {triangleSuper}});
        }
        profiler::stopSection(0, profiler::INIT);
        profiler::stopBranch(0);

        // Invoke replace
        int depth = 1;
        while(!pointsEnc.empty()){
            pair<int, unordered_set<shared_ptr<TriangleB>>> pairEnc = *pointsEnc.begin();
            int i = pairEnc.first;
            unordered_set<shared_ptr<TriangleB>> trianglesIEnc = pairEnc.second;
            pointsEnc.erase(i);

            profiler::startBranch(depth);

            // Prepare edge list for triangle replacement
            profiler::startSection(depth, profiler::PREP);
            unordered_set<Edge> edgesReplacing;
            for(const shared_ptr<TriangleB>& t : trianglesIEnc){
                unordered_set<Edge> edges = t->getEdges();
                edgesReplacing.insert(edges.begin(), edges.end());
            }
            profiler::stopSection(depth, profiler::PREP);

            // Delete & replace all the triangles that i is encroaching on
            for(const shared_ptr<TriangleB>& t : trianglesIEnc){
                // Replace t
                profiler::startSection(depth, profiler::REPLACE);
                vector<shared_ptr<TriangleB>> trianglesNew = replace(pointsS, edgeLinks, edgesReplacing, *t, i);
                profiler::stopSection(depth, profiler::REPLACE);

                profiler::startSection(depth, profiler::SAVE);
                for(const shared_ptr<TriangleB>& tn : trianglesNew){
                    trianglesOutput.insert(tn);
                    // This new triangle isn't complete, populate the global encroaching points list
                    for(int ine : tn->pointsEnc){
                        if(pointsEnc.count(ine)) pointsEnc.find(ine)->second.insert(tn);
                    }
                    // Link edges of new triangle
                    for(Edge edge : tn->getEdges()){
                        edgeLinks.insert({edge, tn});
                    }
                }
                profiler::stopSection(depth, profiler::SAVE);
                profiler::startSection(depth, profiler::REDACT);
                // Delete t and unlink all points
                trianglesOutput.erase(t);
                for(int iOld : t->pointsEnc){
                    if(pointsEnc.count(iOld)) pointsEnc.find(iOld)->second.erase(t);
                }
                profiler::stopSection(depth, profiler::REDACT);
            }

            profiler::stopBranch(depth);
            depth = (int)((1.0f - ((float)pointsEnc.size() / (float)points.size())) * 19.0f) + 1;
        }

        // Convert to other triangle object
        profiler::startBranch(20);
        profiler::startSection(20, profiler::POST);
        vector<Triangle> output;
        for(const shared_ptr<TriangleB>& t : trianglesOutput){
            // Omit super triangle
            TriangleB triangle = *t;
            if(triangle.i1 < indices.size() && triangle.i2 < indices.size() && triangle.i3 < indices.size()) {
                output.push_back({triangle.i1, triangle.i2, triangle.i3});
            }
        }
        profiler::stopSection(20, profiler::POST);
        profiler::stopBranch(20);
        return output;
    }

}