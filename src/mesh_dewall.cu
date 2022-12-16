#include "profiler_mesh.cuh"

class MeshDewall : public Mesh{

private:
    // Attempt to find a point to pair with the supplied Edge to complete a triangle
    int findPoint(const Circles *circles, bool *output,
                  const double *pxLocal, const double *pyLocal, int pointsLocalSize,
                  const double pxEdge1, const double pyEdge1, const double pxEdge2, const double pyEdge2,
                  const int *indicesLocal, const int indexEdge1, const int indexEdge2) const {
        int indexP3 = -1;
        for(int i = 0; i < pointsLocalSize; i++){
            if(indicesLocal[i] != indexEdge1 && indicesLocal[i] != indexEdge2 &&
                isAboveEdge(pxEdge1, pyEdge1, pxEdge2, pyEdge2, pxLocal[i], pyLocal[i])){
                indexP3 = i;
                break;
            }
        }
        if(indexP3 != -1){
            bool valid;
            do{
                valid = true;
                circles->run(output, {pxEdge2, pyEdge2}, {pxEdge1, pyEdge1}, {pxLocal[indexP3], pyLocal[indexP3]});
                for(int i = 0; i < pointsLocalSize; i++){
                    if(output[i] && i != indexP3 &&
                       isAboveEdge(pxEdge1, pyEdge1, pxEdge2, pyEdge2, pxLocal[i], pyLocal[i]) &&
                       indicesLocal[i] != indexEdge1 && indicesLocal[i] != indexEdge2){
                        valid = false;
                        indexP3 = i;
                        break;
                    }
                }
            }while(!valid);
        }

        return indexP3;
    }

    bool edgeFindAndErase(Edge edge, Edge *edges, int& sizeEdges) const {
        bool found = false;
        for(int i = 0; i < sizeEdges; i++){
            if(edges[i].i1 == edge.i1 && edges[i].i2 == edge.i2) found = true;
            if(found && i < sizeEdges - 1) edges[i] = edges[i + 1];
        }
        if(found) sizeEdges--;
        return found;
    }

    // Algorithm source: https://doi.org/10.1016/S0010-4485(97)00082-1
    void triangulateRecursive(Triangle *connections, int &connectionsSize, const double *px, const double *py, int pointsSize,
                              const int *indicesLocal, int indicesLocalSize, const Bounds bounds, const int depth,
                              Edge *edgesActive, int sizeEdgesActive) const {
        profiler_mesh::startBranch(depth);

        // Calculate dividing wall
        profiler_mesh::startSection(depth, profiler_mesh::WALL);
        Wall wall = Wall::build(bounds, depth);
        profiler_mesh::stopSection(depth, profiler_mesh::WALL);

        // Divide points by wall side
        profiler_mesh::startSection(depth, profiler_mesh::DIVIDE_POINTS);
        int sizeIndicesLeft = 0, sizeIndicesRight = 0;
        for(int index = 0; index < indicesLocalSize; index++){
            int i = indicesLocal[index];
            if(wall.side(px[i], py[i])) sizeIndicesRight++;
            else sizeIndicesLeft++;
        }
        int indexIndicesLeft = 0, indexIndicesRight = 0;
        int *indicesLeft = new int[sizeIndicesLeft];
        int *indicesRight = new int[sizeIndicesRight];
        auto *pxLeft = new double[sizeIndicesLeft];
        auto *pyLeft = new double[sizeIndicesLeft];
        auto *pxRight = new double[sizeIndicesRight];
        auto *pyRight = new double[sizeIndicesRight];
        auto *pxLocal = new double[indicesLocalSize];
        auto *pyLocal = new double[indicesLocalSize];
        for(int index = 0; index < indicesLocalSize; index++){
            int i = indicesLocal[index];
            if(wall.side(px[i], py[i])){
                indicesRight[indexIndicesRight] = i;
                pxRight[indexIndicesRight] = px[i];
                pyRight[indexIndicesRight] = py[i];
                indexIndicesRight++;
            }else{
                indicesLeft[indexIndicesLeft] = i;
                pxLeft[indexIndicesLeft] = px[i];
                pyLeft[indexIndicesLeft] = py[i];
                indexIndicesLeft++;
            }
            pxLocal[index] = px[i];
            pyLocal[index] = py[i];
        }
        profiler_mesh::stopSection(depth, profiler_mesh::DIVIDE_POINTS);

        // Initialize with first edges
        if(depth == 0){
            profiler_mesh::startSection(depth, profiler_mesh::FIRST_EDGE);
            // Find nearest point to dividing line (first edge point)
            int aIndex = indicesLocal[0];
            double aDistance = wall.distance(px[indicesLocal[0]], py[indicesLocal[0]]);
            for(int index = 0; index < indicesLocalSize; index++){
                int i = indicesLocal[index];
                double iDistance = wall.distance(px[i], py[i]);
                if(iDistance < aDistance){
                    aIndex = i;
                    aDistance = iDistance;
                }
            }
            bool aSide = wall.side(px[aIndex], py[aIndex]);

            // Find nearest point to A on other side of dividing wall (second edge point)
            int bIndex = aSide ? indicesLeft[0] : indicesRight[0];
            double bDistance = distance(px[aIndex], py[aIndex], px[bIndex], py[bIndex]);
            for(int index = 0; index < (aSide ? sizeIndicesLeft : sizeIndicesRight); index++){
                int i = (aSide ? indicesLeft : indicesRight)[index];
                double iDistance = distance(px[aIndex], py[aIndex], px[i], py[i]);
                if(iDistance < bDistance){
                    bIndex = i;
                    bDistance = iDistance;
                }
            }

            edgesActive = new Edge[2];
            sizeEdgesActive = 2;
            edgesActive[0] = {aIndex, bIndex};
            edgesActive[1] = {bIndex, aIndex};
            profiler_mesh::stopSection(depth, profiler_mesh::FIRST_EDGE);
        }

        profiler_mesh::startSection(depth, profiler_mesh::INIT_LISTS);
        int sizeEdgesActiveWall = 0, sizeEdgesActive1 = 0, sizeEdgesActive2 = 0;
        Edge *edgesActiveWall = new Edge[3 * indicesLocalSize];
        Edge *edgesActive1 = nullptr;
        Edge *edgesActive2 = nullptr;
        profiler_mesh::stopSection(depth, profiler_mesh::INIT_LISTS);

        // Divide inherited active edges
        profiler_mesh::startSection(depth, profiler_mesh::INIT_INHERIT_EDGES);
        for(int i = 0; i < sizeEdgesActive; i++){
            Edge edge = edgesActive[i];
            if(wall.intersects(px[edge.i1], py[edge.i1], px[edge.i2], py[edge.i2])){
                edgesActiveWall[sizeEdgesActiveWall++] = edge;
            }else if(wall.side(px[edge.i1], py[edge.i1])){
                if(edgesActive2 == nullptr) edgesActive2 = new Edge[3 * indicesLocalSize];
                edgesActive2[sizeEdgesActive2++] = edge;
            }else{
                if(edgesActive1 == nullptr) edgesActive1 = new Edge[3 * indicesLocalSize];
                edgesActive1[sizeEdgesActive1++] = edge;
            }
        }
        profiler_mesh::stopSection(depth, profiler_mesh::INIT_INHERIT_EDGES);

        // For all active edges, attempt to complete a triangle and update the active edges list

        Circles *circles = new CirclesKernelShotgun();//new CirclesKernelShotgun();
        //if(indicesLocalSize < 1000) circles = new CirclesSerialPure();
        //else circles = new CirclesSerialShotgun();

        profiler_mesh::startSection(depth, profiler_mesh::PREP);
        circles->initialize(pxLocal, pyLocal, indicesLocalSize);
        bool *circlesOutput = new bool[indicesLocalSize];
        profiler_mesh::stopSection(depth, profiler_mesh::PREP);

        bool edgeActiveInherited = false;
        Edge edgeActiveNext = {0, 0};
        while(edgeActiveInherited || sizeEdgesActiveWall != 0){
            Edge edge;
            if(edgeActiveInherited){
                edge = edgeActiveNext;
                edgeActiveInherited = false;
            }else edge = edgesActiveWall[--sizeEdgesActiveWall];

            profiler_mesh::startSection(depth, profiler_mesh::LOCATE);
            int indexP3 = indicesLocal[findPoint(circles, circlesOutput, pxLocal, pyLocal, indicesLocalSize, px[edge.i1], py[edge.i1], px[edge.i2], py[edge.i2], indicesLocal, edge.i1, edge.i2)];
            profiler_mesh::stopSection(depth, profiler_mesh::LOCATE);

            if(indexP3 > -1){ // Check if we've made a new triangle
                // Add new triangle to output list
                profiler_mesh::startSection(depth, profiler_mesh::SAVE);
                connections[connectionsSize++] = {edge.i1, indexP3, edge.i2};
                profiler_mesh::stopSection(depth, profiler_mesh::SAVE);

                // Update active edges based on the new triangle
                profiler_mesh::startSection(depth, profiler_mesh::CHAIN);
                Edge *edgesPair = new Edge[2];
                edgesPair[0] = {indexP3, edge.i2};
                edgesPair[1] = {edge.i1, indexP3};
                for(int i = 0; i < 2; i++){
                    Edge e = edgesPair[i];
                    if(wall.intersects(px[e.i1], py[e.i1], px[e.i2], py[e.i2])){
                        if(!edgeFindAndErase(e.reverse(), edgesActiveWall, sizeEdgesActiveWall)){
                            edgeActiveInherited = true;
                            edgeActiveNext = e;
                        }
                    }else if(wall.side(px[e.i1], py[e.i1])){
                        if(edgesActive2 == nullptr || !edgeFindAndErase(e.reverse(), edgesActive2, sizeEdgesActive2)){
                            if(edgesActive2 == nullptr) edgesActive2 = new Edge[3 * indicesLocalSize];
                            edgesActive2[sizeEdgesActive2++] = e;
                        }
                    }else{
                        if(edgesActive1 == nullptr || !edgeFindAndErase(e.reverse(), edgesActive1, sizeEdgesActive1)){
                            if(edgesActive1 == nullptr) edgesActive1 = new Edge[3 * indicesLocalSize];
                            edgesActive1[sizeEdgesActive1++] = e;
                        }
                    }
                }
                delete[] edgesPair;
                profiler_mesh::stopSection(depth, profiler_mesh::CHAIN);
            }
        }

        profiler_mesh::startSection(depth, profiler_mesh::PREP);
        circles->cleanup();

        delete circles;
        delete[] circlesOutput;
        profiler_mesh::stopSection(depth, profiler_mesh::PREP);

        profiler_mesh::stopBranch(depth);

        // Recursively call delaunay triangulation on both sides of the wall
        if(edgesActive1 != nullptr){
            Bounds bounds1(bounds.xMin, wall.horizontal ? bounds.xMid : bounds.xMax, bounds.yMin, (!wall.horizontal) ? bounds.yMid : bounds.yMax);
            triangulateRecursive(connections, connectionsSize, px, py, pointsSize, indicesLeft, sizeIndicesLeft, bounds1, depth + 1, edgesActive1, sizeEdgesActive1);
        }
        if(edgesActive2 != nullptr){
            Bounds bounds2(wall.horizontal ? bounds.xMid : bounds.xMin, bounds.xMax, (!wall.horizontal) ? bounds.yMid : bounds.yMin, bounds.yMax);
            triangulateRecursive(connections, connectionsSize, px, py, pointsSize, indicesRight, sizeIndicesRight, bounds2, depth + 1, edgesActive2, sizeEdgesActive2);
        }

        delete[] edgesActiveWall;
        delete[] edgesActive1;
        delete[] edgesActive2;

        delete[] indicesLeft;
        delete[] indicesRight;

        delete[] pxLeft;
        delete[] pyLeft;
        delete[] pxRight;
        delete[] pyRight;
        delete[] pxLocal;
        delete[] pyLocal;
    }

public:
    void run(Triangle *connections, int &connectionsSize, const double *px, const double *py, int pointsSize) const override {
        Bounds bounds(0.0f, 1.0f, 0.0f, 1.0f);

        int *indices = new int[pointsSize];
        for(int i = 0; i < pointsSize; i++) indices[i] = i;

        triangulateRecursive(connections, connectionsSize, px, py, pointsSize, indices, pointsSize, bounds, 0, nullptr, 0);

        delete[] indices;
    }

    string getFileName() const override {
        return "dewall";
    }

    vector<profiler_mesh::Section> getProfilerSections() const override {
        return profiler_mesh::sectionsMeshDewall;
    }

};