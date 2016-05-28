function m = mGeomean(x)

[r,n] = size(x);

%-- If the input is a row, make sure that N is the number of elements in X.
if r == 1, 
    r = n;
   x = x';
   n = 1; 
end

if any(any(x < 0))
    error('The data must all be non-negative numbers.')
end

m = zeros(1,n);
k = find(sum(x == 0) ==0);
m(k) = exp(sum(log(x(:,k))) ./ r);
