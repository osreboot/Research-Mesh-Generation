close all hidden
clc
clear all

points = cast(rand(100000, 2), "single");
writematrix(points, "points.dat", "Delimiter", ' ');

%%
points = readmatrix("points.dat", "Delimiter", ' ');
load("connections.dat", "connections", "-ASCII");
connections = connections + 1;

%%
tic
connectionsMatlab = delaunay(points);
toc

%%
subplot(1, 2, 1);
hold on;
title("MATLAB - Triangles: " + size(connectionsMatlab, 1));
%scatter(points(:, 1), points(:, 2));
trimesh(connectionsMatlab, points(:, 1), points(:, 2));

subplot(1, 2, 2);
hold on;
title("Custom - Triangles: " + size(connections, 1));
%scatter(points(:, 1), points(:, 2));
trimesh(connections, points(:, 1), points(:, 2));