function [y, iter] = newton(f, fp, x0, tol, max)
iter=1; % We start on the first interation
if (feval(fp,x0)==0)
    % Check to see before we get div by 0 erro
    error('Derivative of function is 0')
end
x1 = x0-feval(f,x0)/feval(fp,x0); % The first Newton step
while(abs((x1-x0)/x0)>tol)
    x0=x1; % update the guess
    if (feval(fp,x0)==0)
        % Check to see before we get div by 0 erro
        error('Derivative of function is 0')
    end
    x1=x0-feval(f,x0)/feval(fp,x0); % next iteration
    iter=iter+1;
    if (iter>=max)
        error('Maximum iterations exceeded')
    end
end
    y=x1; % return solution
return
