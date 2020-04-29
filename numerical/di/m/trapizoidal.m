function [val] = trapizoidal(func, a, b, N)

val = 0;
xx = linspace(a, b, N+1);
dx = xx(2) - xx(1);
for i=2:length(xx)-1
    xx1=xx(i);
    val=val+dx*double(func(xx1));
end
val = val + dx*(0.5*double(func(xx(1))) + 0.5*double(func(xx(end))));

end