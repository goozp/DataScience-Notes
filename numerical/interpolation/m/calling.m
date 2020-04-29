clc; clear all; close all;

% approximate f(0.18)
n = 3; % 4 points, 3+1=4
x = [0.1, 0.2, 0.3, 0.4]; % x
Q = [-0.62049958, -0.28398668, 0.00660095, 0.24842440]; % f(x)
xx = 0.18;

% calling Neville's Method
y = neville(xx, n, x, Q);
fprintf('Approximated f(0.18) = %f \n', y);

[a, b, c, d] = cubicSpline(n, x, Q);
y2 = a(1) + b(1)*0.08 + d(1)*0.08^3;
fprintf('Approximated f(0.18) = %f \n', y2);

