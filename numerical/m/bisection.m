function [m, err, i] = bisection(f, a, b, N, tol)
%% Bisection Method
%% Inputs: 
%	The root-finding problem f(x)=0;
%	Left and right end points a, b;
%	Max # of iterations N;
%	Tolerance tol;
%% Outputs:
%    Approximation m;
%    Error err;
%    # of iterations N;
%%
disp('------------------------------------------------------------------------')
disp("Bisection Method")
disp('Iter     a          b           pn       Abs_err       Rel_err');
disp('------------------------------------------------------------------------')

err = 1;
FA = f(a); 
lastP = 0;
for i = 1:N
    m = (a+b)/2;
    FP = f(m);
    err = abs((b-a)/2);
    if lastP ~= 0
        Rel_err = num2str(abs((m - lastP)/m), '%0.15f');
    else
        Rel_err = '\';
    end
    fprintf('%d \t %f \t %f \t %f \t %f \t %s \n', i, a, b, m, err, Rel_err); 
    lastP = m;
    if err < tol
        break;
    elseif FA*FP>0
        a = m; 
        FA = FP;
    else 
        b = m;
    end
end

if err > tol
    fprintf('The method failed to converge within iteration N = %d', i);
end

end