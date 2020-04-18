function [p, err, i] = fixedpoint(g, p0, N, tol)
%% Fixed-point Iteration
%% Inputs: 
%   The fixed-point problem g(x)=x;
%	Initial approx p0;
%	Max # of iterations N;
%	Tolerance tol;
%% Outputs: 
%   Approximation p;
%	Error (Relative error) err;
%   # of iterations N;
%%
disp('----------------------------------------------------------------')
disp('Fixed-point Iteration')
disp('Iter   p(n-1)         pn         err         Rel_err')
disp('----------------------------------------------------------------')
fprintf('%d \t %s \t %f \t %s \t %s \n', 0, '\', p0, '\', '\')
    
for i = 1:N
    p = g(p0);
    err = abs(p-p0);
    Rel_err = abs((p-p0)/p);
    fprintf('%d \t %f \t %f \t %f \t %0.15f \n', i, p0, p, err, Rel_err)
    p0 = p;
    if err < tol
        break;
    end  
end

if err > tol
    fprintf('The method failed to converge within iteration N = %d', i);
end

end