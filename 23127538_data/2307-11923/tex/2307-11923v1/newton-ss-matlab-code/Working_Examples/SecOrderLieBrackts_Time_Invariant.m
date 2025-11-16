%% 

%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms z1 z2 d v w w0 a h H F_star w_d t c;


F(z1,z2) = F_star -1/2 * H *((z1)^2 + (z2)^2);



% theta = w0 * t;
k = 1;

J0 = [0, w0;-w0,0];                     



b0 = [J0*[z1;z2]; w_d; h*(-v   + F(z1,z2))];
 
b1 =   exp(d)* (-v   + F(z1,z2)) *  [0;...
                                   1;...
                                            0;...
                                            0];

b2 = a *[0;...
         1;...
         0;...
         0];
                                                                                                         
b3 = [ 0;...
       0;...
      -exp(d) * 8 * 1/(a)^2 * w_d *  (-v   + F(z1,z2));...
       0 ];

%% Inputs

p = 0.9;

u1 = w^(1-p) * sin(k* w * t);
u2 = w^(p) * cos(k*w* t);
u3 = w^(2 - 2*p) * cos(2 * k * w *t) ;

%  
% LieBracketSys =
%  
%  -(H*a*b*d*cos(t*w0)*(x*cos(t*w0) + y*sin(t*w0)))/2
%  -(H*a*b*d*sin(t*w0)*(x*cos(t*w0) + y*sin(t*w0)))/2
%                 -h*(v - F_star + (H*(x^2 + y^2))/2)
%                                    -d*w_d*(H*d - 1)

%% 
B = [b1,b2,b3];

U = [u1,u2,u3];

var = [z1,z2,d,v];




[LBS,LBSlim] = secondOrdLBS(B,U,var,k);

LieBracketSys = b0 + LBSlim


aa = sum(B(1,:).*U); 
bb = sum(B(2,:).*U);
cc = sum(B(3,:).*U);
dd = sum(B(4,:).*U);

% ee = sum(B(5,:).*U);


gg = [aa;bb;cc;dd];

NominalSys = b0 + simplify(gg)

 