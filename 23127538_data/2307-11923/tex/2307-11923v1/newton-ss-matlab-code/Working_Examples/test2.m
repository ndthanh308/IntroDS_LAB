%%
clc;
clear;

syms x y H w0 a b h wd;
assume(x,'real');
assume(y,'real');
assume(H > 0);

xdot = (1-1/H * y * exp(x));
ydot = -(y-H);

V = 1/2 * (y-H)^2 + (1-1/H * y * exp(x))^2;

Vdot = gradient(V,[x,y]).' * [xdot;ydot];

% expand(Vdot)

A = [0,w0,0;-w0,-a*b*H/2,0;0,0,-h]
A_new = [0,w0,0,0;-w0,-a*b/2,0,0;0,0,-wd,0;0,0,0,-h]
% A = [0,1;-1,-a*H/2];
