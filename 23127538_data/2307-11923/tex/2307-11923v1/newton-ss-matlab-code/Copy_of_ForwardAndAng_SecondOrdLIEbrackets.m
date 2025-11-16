%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms x y d z F(x,y) h k w w0 t p w_d w_z p0 p1 p2 p3 p4 p5 c1 c2 ga ga2 q q_hess;


assume([x y d z F(x,y) h k w w0 t p w_d w_z p1 p2 p3 p4 c1 c2 ga ga2 q q_hess],'real')
% h = 1/w;

F(x,y) =  -1/2 * q_hess*(x^2 + y^2);


p1 = 0.3;
p2 = 0.7;

p3 = 0.6;


% p1 = 0.5;
% 
% p2 = 0.7;
% 
% 
% p4 = (4*p1 - p2 -1);
% 
% p3 = p2 - 2*p1;

% k = w^p4;

% p3 = 2*1.5 - 1;
% c2 = 2;
% h = 1/(w^p3);
% w0 = 1/w^p4;
% k = w^p5;
c1 = 2;
% q = w^(0.15);
b0 = [0;0;w_d*(d - d^2 * z);-w_z *z ];

% ga = w^(-1/2);

% q = sqrt(w);
% ga2 = w;
% ga = k*w^(1/2);

% q = w^(0.15);
% w0 = (1/w)^(p4);
theta = w0  * t;
syms al be;
b1 =    d * F(x,y) *  [cos(theta);...
                       sin(theta);...
                          0;...
                          0];

b2 =     q * [cos(theta);...
              sin(theta);...
                            0;...
                            0];
  
% b2 = [1;1;0;0];

b3 = [ 0;...
       0;...
       0;...
      8 * k^2 * 1/q^2 *  w_z * F(x,y)];
    
% b4 = [ 0;...
%        0;...
%        0;...
%        -c1 * w_z *  1/ga  * sin(h* F(x, y)) ];

% p = 1;


% p1 = 1.5;
u1 = w^p1 * sin(k* w * t);
u2 = w^p2 * cos(k * w * t);


% p1 = 1/2;
% p1 = 1/2;
% p2 = 2  - 2*p1;

u3 = w^p3 * cos(2 * k * w *t) ;
% u4 = w^p2 * sin(2*k*w*t);
% u4 = w^(2-2*p) * sin(  k * w * t);



%% 
B = [b1,b2,b3];

U = [u1,u2,u3];

var = [x,y,d,z];


% [LBS,LBSlim] = secondOrdLBS(B,U,var)

[LBS,LBSlim] = secondOrdLBS(B,U,var,k);

LieBracketSys = b0 + LBSlim
% 
% aa = sum(B(1,:).*U); 
% bb = sum(B(2,:).*U);
% cc = sum(B(3,:).*U);
% 
% gg = [aa;bb;cc];
% 
% NominalSys = b0 + simplify(gg);

%% 
aa = LBS(1);
bb = LBS(2); 
AA = children(expand(aa));
GG = children(expand(bb));

for i = 1:length(AA)

BB(i,:) = AA(i);

end

for i = 1:length(GG)

CC(i,:) = GG(i);

end

    
% BB
% CC
%     
cc = LBS(4);

GD = children(expand(cc));
for i = 1:length(GD)

DD(i,:) =  GD(i);
end

% DD
%    
% sum( limit(DD,w,inf))
 
 