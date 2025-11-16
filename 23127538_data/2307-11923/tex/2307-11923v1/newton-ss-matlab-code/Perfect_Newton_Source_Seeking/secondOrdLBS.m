function [LBS,LBSlim,LBS_first_order] = secondOrdLBS(B, U, var,k)
%SECONDORDLBS Calculates the second order Lie Bracket Average of a input
%affine system with periodic inputs
%   B - matrix of vector fields
%   U - vector of inputs
%   p - syms variables of the LBS

syms w t;

assume(w,'positive')

assume(t,'positive')

m = size(U,2);
n = size(B,1);

L_BiBj = zeros(n,nchoosek(m,2));

Gamma_ij = zeros(1,nchoosek(m,2));

L_BiBjGamma_ij = zeros(n,1);

for i = 1:(m-1)
    for j = (i+1):m
        
        
%       L_BiBj(:,ic)  =   lie_brackets(B(:,i),B(:,j)) * gamma_ij(U(i),U(j));
        
        L_BiBjGamma_ij = L_BiBjGamma_ij + lie_brackets(B(:,i),B(:,j),var) * gamma_ij(U(i),U(j),k);
       
    end
end

L_BiBjGamma_ij = simplify(L_BiBjGamma_ij);


LBS_first_order = limit(L_BiBjGamma_ij,w,inf);
%% Second Order Lie Brackets

m = size(U,2);
n = size(B,1);

L_BiBjBmGamma_ijm = zeros(n,1); 

count = 1;
for i = 1:(m-1)
    for j = (i+1):m
        for z = 1:m
            
            
            
  L_BiBjBmGamma_ijm = L_BiBjBmGamma_ijm + lie_brackets(lie_brackets(B(:,i),B(:,j),var),B(:,z),var)*gamma_ijm(U(i),U(j),U(z),k);
            
% xx  = limit(lie_brackets(lie_brackets(B(:,i),B(:,j),var),B(:,z),var)*gamma_ijm(U(i),U(j),U(z)),w,inf);
% % %    xx  = lie_brackets(lie_brackets(B(:,i),B(:,j),var),B(:,z),var)*gamma_ijm(U(i),U(j),U(z));
% % %            
%        if  sum(xx) ~= 0
%            yy(count,:) = [i,j,z];
%            X(:,count) =  xx;
%            count = count + 1;
%        end
        
       end
    end
end

%% 
% a = X(4,1);
% 
% b = X(4,2);

%  an = a - b;

%% 
LieBracketSystem = L_BiBjGamma_ij + L_BiBjBmGamma_ijm;



LBS= simplify(LieBracketSystem);



LBSlim = limit(LBS,w,inf);

end

