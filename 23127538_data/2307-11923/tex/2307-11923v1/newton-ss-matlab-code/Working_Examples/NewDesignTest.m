%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms x y z_e d z F(x,y) F_star h k w w0 t p w_d w_z p0 p1 p2 p3 p4 p5 c1 c2 ga ga2 q q_hess x_star y_star;


assume([x y d z F(x,y) h k w w0 t p w_d w_z p1 p2 p3 p4 c1 c2 ga ga2 q q_hess],'real')
% h = 1/w;

%  F(x,y) = F_star -1/2 * q_hess*((x-x_star)^2 + (y-y_star)^2);




p2 = 0.5;
p1 = 1-p2;

% p3 = 2 - 2*p2;




c1 = 2;
% q = w^(0.15);
% b0 = [0;0;-z_e * h  + F(x,y); w_d*(d - d^2 * z);-w_z *z ];

% ga = w^(-1/2);

% q = sqrt(w);
% ga2 = w;
% ga = k*w^(1/2);

% q = w^(0.15);
% w0 = (1/w)^(p4);

theta = w0 * t;

u1 =   sqrt(w) *  (  sin(F(x,y)/w) + q);                    
b1 =  u1 *   [1;...
              0];

b2 =    u1 * [0;...
              1];                        
                        
                        
  


% b3 = [ 0;...
%        0;...
%        0;...
%        0;...
%       8 * k^2 * 1/q^2 *  w_z * (-z_e * h  + F(x,y))];
    
% b4 = [ 0;...
%        0;...
%        0;...
%        -c1 * w_z *  1/ga  * sin(h* F(x, y)) ];

% p = 1;


% p1 = 1.5;
u1 = w^p1 * cos(k* w * t);
u2 = w^p2 * sin(k * w * t);


% p1 = 1/2;
% p1 = 1/2;
% p2 = 2  - 2*p1;

% u3 = w^p3 * cos(2 * k * w *t) ;
% u4 = w^p2 * sin(2*k*w*t);
% u4 = w^(2-2*p) * sin(  k * w * t);



%% 
B = [b1,b2];

U = [u1,u2];

var = [x,y];


% [LBS,LBSlim] = secondOrdLBS(B,U,var)

[LBS,LBSlim, LBSfirst] = secondOrdLBS(B,U,var,k);

LieBracketSys =  LBSfirst
% 
% aa = sum(B(1,:).*U); 
% bb = sum(B(2,:).*U);
% cc = sum(B(3,:).*U);
% 
% dd = sum(B(4,:).*U);
% ee = sum(B(5,:).*U);
% 
% 
% gg = [aa;bb;cc;dd;ee];
% 
% NominalSys =  simplify(gg);

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
% cc = LBS(5);
% 
% GD = children(expand(cc));
% for i = 1:length(GD)
% 
% DD(i,:) =  GD(i);
% end

% DD
%    
% sum( limit(DD,w,inf))
 
 