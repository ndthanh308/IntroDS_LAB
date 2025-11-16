%% 

%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms z1 z2 d v w w0 a h H F_star w_d t c ps;


F(z1,z2) = F_star -1/2 * H *((z1)^2 + (z2)^2);



% theta = w0 * t;
k = 1;

J0 = [0, w0;-w0,0];                     



f0 = [J0*[z1;z2]; h*(-v   + F(z1,z2))];
 
f1 =    (-v   + F(z1,z2)) *  [0;...
                              1;...
                              0];

f2 = a *[0;...
         1;...
         0];
                                                                                                         


%% Inputs

p = ps;

u1 = w^(1-p) * sin(k* w * t);
u2 = w^(p) * cos(k*w* t);


%  
% LieBracketSys =
%  
%  -(H*a*b*d*cos(t*w0)*(x*cos(t*w0) + y*sin(t*w0)))/2
%  -(H*a*b*d*sin(t*w0)*(x*cos(t*w0) + y*sin(t*w0)))/2
%                 -h*(v - F_star + (H*(x^2 + y^2))/2)
%                                    -d*w_d*(H*d - 1)

%% 
B = [f1,f2];

U = [u1,u2];

var = [z1,z2,v];




[lbs_fo,gamma_fo,lbs_so,gamma_so,LBS,LBSlim] = secondOrdLBS(B,U,var,k);

LieBracketSys = f0 + LBSlim


aa = sum(B(1,:).*U); 
bb = sum(B(2,:).*U);
cc = sum(B(3,:).*U);
% dd = sum(B(4,:).*U);

% ee = sum(B(5,:).*U);


gg = [aa;bb;cc];

NominalSys = f0 + simplify(gg)

 