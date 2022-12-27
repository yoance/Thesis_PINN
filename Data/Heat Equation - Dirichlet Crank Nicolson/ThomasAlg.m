function sol = ThomasAlg(a_old,b,c,r)
% vector lengths
% b = m
% a = c = m-1
% b = [2;2;2]; a = [-1;-1]; c = a; r = [1;2;3];

m = length(b);

c(1) = c(1)/b(1);
r(1) = r(1)/b(1);


% however, since a starts from the second row while a's length is only m-1,
a = zeros(m,1);
a(2:end) = a_old;

% reducing rows 2 to m-1
for j = 2:m-1
    b(j) = b(j)-a(j)*c(j-1);
    r(j) = r(j)-a(j)*r(j-1);
    c(j) = c(j)/b(j);
    r(j) = r(j)/b(j);
end

% reducing last row
% however, since a starts from the second row while a's length is only m-1,
% we must change the index here to m-1
b(m) = b(m) - a(m)*c(m-1);
r(m) = r(m) - a(m)*r(m-1);
r(m) = r(m)/b(m);

% backwards substitution to solve system
for j = m-1:-1:1
    r(j) = r(j) - c(j)*r(j+1);
end

sol = r;
end