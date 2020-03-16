function  m = bisection(f, a, b, tol)
% Bisection Method
y1 = feval(f, a);
y2 = feval(f, b);
i = 0; 

if y1 * y2 > 0
   disp('Have not found a change in sign. Will not continue...');
   m = 'Error';
   return
end 

disp('Iter    a          b            xn');
while (abs(b - a) >= tol)
    i = i + 1;
    m = (b + a)/2;
    y3 = feval(f, m);
    if y3 == 0
        fprintf('Root at x = %f \n\n', m);
        return
    end
    fprintf('%2i \t %f \t %f \t %f \n', i-1, a, b, m);   

    if y1 * y3 > 0
        a = m;
        y1 = y3;
    else
        b = m;
    end
end 
