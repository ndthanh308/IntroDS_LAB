clear;
clc;

syms k r y d z F(r) w t w_z w_d h p ga q;

assume(k,'positive')

assume(w,'positive')


% h = 1/w;
% k = 1/sqrt(w);

b0 = [0;0;w_d * (d -  d^2 * z);0];



% syms theta; 
% 
% b11 = cos(theta - y);
% b11 = expand(b11);
% b12 = 1/r * sin(theta - y);
% b12 = expand(b12);
% 
% b11 = subs(b11,theta, k*w*t - h*F(r));
% b12 = subs(b12,theta, k*w*t - h*F(r));
% 
% b11 = expand(b11);
% b12 = expand(b12);


% b1 = ga * [cos(y)*cos(h*F(r)) - sin(y)*sin(h*F(r));...
%      - 1/r  * ( cos(h*F(r))*sin(y) + cos(y)*sin(h*F(r)) );...
%       0;...
%       0];
% 
% b2 = ga *  [cos(y)*sin(h*F(r)) + cos(h*F(r))*sin(y);...
%      1/r * ( cos(y)*cos(h*F(r)) - sin(y)*sin(h*F(r)) );...
%      0;
%      0];

 syms w0;
 
 theta = w0 *t;
 b1 =   d* F(r)*[cos(y + w0 * t);...
            sin(y + w0*t)/r;...
                       0;...
                       0];
                   
 b2 =   q*[cos(y + w0 * t);...
            sin(y + w0*t)/r;...
                       0;...
                       0];               
 
b1 = simplify(b1);
b2 = simplify(b2);


b3 = [0; 0; 0;  8 * k^2 * 1/q^2 * w_z  * F(r) ];



%% 
p1 = 0.3;
p2 = 0.7;
p3 = 0.6;

% clear u1 u2 u3;

u1 = w^p1 *  sin(k *w * t);

u2 = w^p2 *  cos( k * w * t);

u3 = w^p3*  cos(  2 *  k * w * t)  ;


U = [u1,u2,u3];

%% 
B = [b1,b2,b3];

var = [r,y,d,z];

% 
% [LBS,LBSlim] = secondOrdLBS(B,U,var)

[LBS,LBSlim] = secondOrdLBS(B,U,var,k);

LieBraSys = b0 + LBSlim
