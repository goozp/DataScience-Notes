clc; clear all; close all;

%a = 1;
%b = 2;
%n = 4;
%f = @(t)t.*log(t);
%result = trapezoidal(f, a, b, n);

%fprintf('The following integrals is %f \n', result);



f = @(t)(1/(sqrt(2*pi)))*exp(-(t.^2)/2);
N = 20;
xx = 0.5;
err = 10^-5;
error = 1;
while error > err
    yy = simpson(f, 0, xx, N) - 0.45;
    x2 = double(xx-(yy./f(xx)));
    error = abs(xx-x2);
    xx=x2;
end

fprintf('The Root of f(x) using Newton method with Simpson intergral for x0=0.5 is %f\n', xx)


f = @(t)(1/(sqrt(2*pi)))*exp(-(t.^2)/2);
N = 40;
xx = 0.5;
err = 10^-5;
error = 1;
while error > err
    yy = trapizoidal(f, 0, xx, N) - 0.45;
    x2 = double(xx-(yy./f(xx)));
    error = abs(xx-x2);
    xx=x2;
end

fprintf('The Root of f(x) using Newton method with Trapezoidal intergral for x0=0.5 is %f\n', xx)