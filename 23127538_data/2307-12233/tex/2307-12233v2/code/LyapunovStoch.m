clc

A = [1/3 1/3 1/3;
     0   1/2 1/2;
     1/6 1/6 2/3];

sort(eig(A))

[U,S,V] = svd(A)

P = A'*A
sort(eig(P))

M = P-eye(size(A))
sort(eig(M))

x = [-1 1 3]';

min(A*x)
max(A*x)

max(A*x)-min(A*x)-max(x)+min(x)