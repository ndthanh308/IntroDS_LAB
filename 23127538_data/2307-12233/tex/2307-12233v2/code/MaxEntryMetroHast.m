close all
clear all
clc

% A = [0 1 1 0 0 0;
%      1 0 0 1 0 0;
%      1 0 0 1 0 0;
%      0 1 1 0 1 1;
%      0 0 0 1 0 1;
%      0 0 0 1 1 0];
% A = [0 1 1 0 0;
%      1 0 0 1 0;
%      1 0 0 1 0;
%      0 1 1 0 1;
%      0 0 0 1 0];
 A = [0 1 1 0 0 0;
     1 0 0 1 0 0;
     1 0 0 1 0 0;
     0 1 1 0 1 1;
     0 0 0 1 0 1;
     0 0 0 1 1 0];
d = sum(A);
 
G = graph(A);
plot(G)
n = numnodes(G);
 
P = zeros(n);
for i = 1:n
    for j = 1:i-1
        if A(i,j)
            P(i,j) = 1/(1+max(d(i),d(j)));
            P(j,i) = P(i,j);
        end
    end
end
for i = 1:n
    P(i,i) = 1-sum(P(i,:));
end
P


deltai = zeros(n,1);
for i = 1:n
    Ni = neighbors(G,i);
    deltai(i) = d(i);
    for j_ = 1:d(i)
        j = Ni(j_);
        if deltai(i) < d(j)
            deltai(i) = d(j);
        end
    end
end
delta = min(deltai);
delta

trueMax = max(max(P))
trueMax_off = max(max(P-diag(diag(P))))
UpperBound_delta = delta/(1+delta)
UpperBound_minmax = 1-min(d)/(1+max(d))
UpperBound_cross = min(UpperBound_delta,UpperBound_minmax)
















