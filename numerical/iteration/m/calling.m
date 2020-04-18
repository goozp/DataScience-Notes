clear all; close all; clc;

tol = .0001;
f = @(x)(sqrt(x)-cos(x));

% Bisection Method
a = 0;
b = 1.0;
[p1, err, i1] = bisection(f, a, b, 100, tol);

% fixed-point
g =  @(x)(cos(x)^2);
tol = .0001;
p0 = 0.5;
[p2, err2, i2] = fixedpoint(g, p0, 1000, tol);

%  newton
syms x
df = matlabFunction(diff(f, x));
p0 = 0.5;
[p3, err3, i3] = newton(f, df, p0, 100, tol);

fprintf("Using bisection method, after %2.0f iteration p = %f \n", i1, p1)
fprintf("Using fixed-point iteration, after %2.0f iteration p = %f \n", i2, p2)
fprintf("Using newton's method, after %2.0f iteration p = %f \n", i3, p3)