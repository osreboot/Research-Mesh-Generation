close all hidden
clc
clear all

points = cast(rand(200000, 2), "single");
writematrix(points, "points.dat", "Delimiter", ' ');
%save("points.dat", "points", "-ASCII");

%%
points = readmatrix("points.dat", "Delimiter", ' ');
%load("points.dat", "points", "-ASCII");
load("connections.dat", "connections", "-ASCII");
connections = connections + 1;

%%
tic
connectionsMatlab = delaunay(points);
toc

subplot(1, 2, 1);
hold on;
title("MATLAB - Triangles: " + size(connectionsMatlab, 1));
%scatter(points(:, 1), points(:, 2));
trimesh(connectionsMatlab, points(:, 1), points(:, 2));

subplot(1, 2, 2);
hold on;
%axis([0.8 0.95 0.7 0.85]);
title("Custom - Triangles: " + size(connections, 1));
%scatter(points(:, 1), points(:, 2));
trimesh(connections, points(:, 1), points(:, 2));