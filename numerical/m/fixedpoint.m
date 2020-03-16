function [ iter ] = fixedpoint(f, x0, tol, n)
iter=0;
u=feval(f, x0);
err=1; % init err as a big number

disp('-------------------------------------------')
disp('Iter    xn          f(xn)          err ')
disp('-------------------------------------------')
fprintf('%2.0f %12.6f %12.6f\n', iter, x0, u) % iter 0

while ( err > tol ) && ( iter <= n )
    x1 = u;
	err = abs(x1 - x0);
	x0 = x1;
	u = feval(f, x0);
	iter = iter + 1;
    fprintf('%2.0f %12.6f %12.6f %12.8f\n', iter, x0, u, err)
end

if(iter > n)
	disp('Method failed to converge')
end
end
