function [lbs_fo_ij, gamma_fo_ij,lbs_so_ijm,gamma_so_ijm,LBS,LBSlim,LBS_first_order] = secondOrdLBS(B, U, var,k)
%SECONDORDLBS Calculates the second order Lie Bracket Average of a input
%affine system with periodic inputs
%   B - matrix of vector fields
%   U - vector of inputs
%   p - syms variables of the LBS
%%
syms w t;

assume(w,'positive')

assume(t,'positive')

m = size(U,2);
n = size(B,1);

L_BiBj = zeros(n,nchoosek(m,2));

Gamma_ij = zeros(1,nchoosek(m,2));

L_BiBjGamma_ij = zeros(n,1);

lbs_fo_ij = cell(m,m);
gamma_fo_ij = sym(zeros(m,m));

for i = 1:(m-1)
    for j = (i+1):m
        
        
%       L_BiBj(:,ic)  =   lie_brackets(B(:,i),B(:,j)) * gamma_ij(U(i),U(j));

        lbs_fo_ij{i,j} =  lie_brackets(B(:,i),B(:,j),var);
        gamma_fo_ij(i,j) = gamma_ij(U(i),U(j),k);

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

lbs_so_ijm = cell(m,m,m);
gamma_so_ijm = sym(zeros(m,m,m));
for i = 1:(m-1)
    for j = (i+1):m
        for z = 1:m
           
        lbs_so_ijm{i,j,z} = lie_brackets(lie_brackets(B(:,i),B(:,j),var),B(:,z),var);
        gamma_so_ijm(i,j,z) = gamma_ijm(U(i),U(j),U(z),k);
  L_BiBjBmGamma_ijm = L_BiBjBmGamma_ijm + lie_brackets(lie_brackets(B(:,i),B(:,j),var),B(:,z),var)*gamma_ijm(U(i),U(j),U(z),k);
                
       end
    end
end



%% 
LieBracketSystem = L_BiBjGamma_ij + L_BiBjBmGamma_ijm;
LBS= simplify(LieBracketSystem);

LBSlim = limit(LBS,w,inf);

end

