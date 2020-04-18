function [p, err, i] = newton(f, fd, p0, N, tol)
%% Newton's Method
%% Inputs: 
%   The root-finding problem f(x)=0;
%	Initial approx p0;
%	ax # of iterations N;
%	Tolerance tol;
%% Outputs: 
%   Approximation p;
%	Error (Relative error);
%	# of iterations N;
%%
disp('---------------------------------------------------------')
disp("Newton's Method")
disp('Iter  p(n-1)        pn        err          Rel_err')
disp('---------------------------------------------------------')
fprintf('%d \t %s \t %f \t %s \t %s \n', 0, '\', p0, '\', '\')

for i = 1:N  
    p = p0 - f(p0)/fd(p0);
    err = abs(p-p0);
    Rel_err = abs((p-p0)/p);
    fprintf('%d \t %f \t %f \t %f \t %0.15f \n', i, p0, p, err, Rel_err)
    if err < tol
        break;
    end  
    p0 = p;
end

if err > tol
    fprintf('The method failed to converge within iteration N = %d', i);
end

end