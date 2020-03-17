function  [i, p] = bisection(f, a, b, tol, N)
% Bisection Method

i = 1;  
fa = feval(f, a);

disp('-------------------------------------------')
disp("Bisection Method")
disp('Iter        a           b            pn');
disp('-------------------------------------------')

while (i <= N)
    p = a + (b-a)/2;
    fp = feval(f, p);
    if fp == 0 || (b-a)/2 < tol
       fprintf('%2i \t %f \t %f \t %f \n', i-1, a, b, p);  
       return 
    end
    
    fprintf('%2i \t %f \t %f \t %f \n', i-1, a, b, p);  
    
    i = i + 1;
    if fa * fp >0
        a = p;
    else
        b = p;
    end 
end
