# What is this?
This research project explores the application of NVIDIA Tensor Core hardware acceleration in two-dimensional surface mesh generation. Two algorithms are implemented: the "DeWall" [1] recursive partitioning algorithm and the "Blelloch" [2] sequential subdivision algorithm. This project has no dependencies besides the standard CUDA libraries.

# Usage
There are several parts to this project:

`test.m` is a MATLAB program that can: 1) generate the point cloud *points.dat* consumed by the CUDA C++ program, 2) generate its own triangulation using built-in MATLAB functions 3) visualize both connected meshes.

`main.cu` is a CUDA C++ program that generates a triangulation of *points.dat* in *connections.at* using either of the above algorithms based on which invocation is uncommented in the `int main()` block.

`profiler.cu` writes the performance statistics of `main.cu` to *profile.csv*.

# Examples
TODO

# References
[1] P. Cignoni, C. Montani and R. Scopigno, "DeWall: A fast divide and conquer Delaunay triangulation algorithm in Ed," Computer-Aided Design, vol. 30, no. 5, pp. 333-341, 1998. 

[2] G. E. Blelloch, Y. Gu, J. Shun and Y. Sun, "Parallelism in Randomized Incremental Algorithms," in SPAA '16: Proceedings of the 28th ACM Symposium on Parallelism in Algorithms and Architectures, Pacific Grove, 2016. 
