%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms x d y z F(x) k w t p w_z;


b1 = [1;...
      0;...
      0;...
      0];

b2 = [0;...
      0;...
      2*k*F(x);...
      0];


b3 = [ 0; 0; 0; -w_z * 8 * k^2  * F(x)];


p = 1/2;

u1 = w^p * sin(k * w * t);

u2 = w^(1-p) * cos(k * w * t);

u3 = w^(2-2*p) * cos(2 * k * w * t);

% u3 = w^p2 * cos(2* k * w * t);
% u4 = w^p2 * sin(2* k * w * t);

%% 
B = [b1,b2,b3];

U = [u1,u2,u3];

p = [x,d,y,z];


LBS = secondOrdLBS(B,U,p)


