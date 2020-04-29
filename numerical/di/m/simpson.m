function [val] = simpson(func, a, b, N)

xx = linspace(a, b, N+1);
dx = xx(2) - xx(1);
val = (dx/3)*(double(func(xx(1))) + double(func(xx(end))));

for i=2:length(xx)-1
    xx1 = xx(i);
    if mod(i, 2)==0
        val = val+(dx/3)*4*double(func(xx1));
    else
        val = val+(dx/3)*2*double(func(xx1));
    end
end

end