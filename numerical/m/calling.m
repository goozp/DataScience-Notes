tol = .0001;
f = @(x)(sqrt(x)-cos(x));

% Bisection Method
a = -1.0;
b = 1.0;
[i1, p1] = bisection(f, a, b, tol, 100);

% fixed-point
g =  @(x)(cos(x)^2);
tol = .0001;
p0 = 0;
[i2, p2] = fixedpoint(g, p0, tol, 1000);

%  newton
syms x
df = matlabFunction(diff(f, x));
p0 = 0.1;
[i3, p3] = newton(f, df, p0, tol, 100);

fprintf("Using bisection method, after %2.0f iteration p = %12.6f \n", i1, p1)
fprintf("Using fixed-point iteration, after %2.0f iteration p = %12.6f \n", i2, p2)
fprintf("Using newton's method, after %2.0f iteration p = %12.6f \n", i3, p3)