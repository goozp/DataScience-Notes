x0 = 0;
n = 100;
tol = .0001;

syms x
f = @(x)(sqrt(x)-cos(x));
g = matlabFunction(diff(f, x));
% Bisection Method
a = -1.0;
b = 1.0;
bisection(f, a, b, tol)
% fixed-point
tol = .0001;
fixedpoint(f, x0, tol, n)
%  newton
newton(f, g, x0, tol, n)