function [ i, p ] = fixedpoint(f, p0, tol, N)
% Fixed-point Iteration

e = 1; % init error
i = 1;

disp('-------------------------------------------')
disp('Fixed-point Iteration')
disp('Iter       pn         f(pn)         err ')
disp('-------------------------------------------')

while (e > tol) && (i <= N)
    p = feval(f, p0);
    e = abs(p - p0);
    
    fprintf('%2.0f %12.6f %12.6f %12.8f \n', i, p0, p, e)
    
    p0 = p;
    i = i + 1;
end
