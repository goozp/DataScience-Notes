function [ i, p ] = newton(f, df, p0, tol, N)
% Newton's Method

i=1; 

disp('-------------------------------------------')
disp("Newton's Method")
disp('Iter       pn          f(pn)')
disp('-------------------------------------------')

while (i <= N)
    p = p0 - feval(f, p0)/feval(df, p0);
    if abs(p - p0) < tol
        fprintf('%2.0f %12.6f %12.6f \n', i, p0, p)
        return
    end 
    fprintf('%2.0f %12.6f %12.6f \n', i, p0, p)
    i = i + 1;
    p0 = p;
end

