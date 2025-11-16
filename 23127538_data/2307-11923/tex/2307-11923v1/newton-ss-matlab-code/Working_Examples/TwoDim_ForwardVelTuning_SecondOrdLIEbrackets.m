%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms x y v d e1 e2 F(x,y) F_star h k w w0 t p w_d w_z p c c2 ga ga2 q H x_star y_star a b;




F(x,y) = F_star -1/2 * H *((x)^2 + (y)^2);



theta = w0 * t;
k = 1;

%% Zhang Algorithm

% b0 = [0;0;h*(-v   + F(x,y))];
% 
% b1 =   b *  (-v   + F(x,y)) *  [cos(theta);...
%                                    sin(theta);...
%                                             0];
% 
% b2 = a *[cos(theta);...
%           sin(theta);...
%                    0];
%                         

%% Ricatti Inverse Estimator:


b0 = [0;0;h*(-v   + F(x,y)); w_d*d];

b1 =   b * d * (-v   + F(x,y)) *  [cos(theta);...
                                   sin(theta);...
                                            0;...
                                            0];

b2 = a *[cos(theta);...
          sin(theta);...
                   0;...
                   0];
                                                                                                         
b3 = [ 0;...
       0;...
       0;...
      -d^2 * 8 * 1/(a)^2 * w_d *  (-v   + F(x,y))];
% 
%   
%% Exponential Inverse Estimator:

% b0 = [0;0;h*(-v   + F(x,y)); w_d];
% 
% b1 =  1/H * exp(d) * (-v   + F(x,y)) *  [cos(theta);...
%                                       sin(theta);...
%                                                0;...
%                                                0];
% 
% b2 = a *  [cos(theta);...
%           sin(theta);...
%                    0;...
%                    0];
%                                                                                                          
% b3 = [ 0;...
%        0;...
%        0;...
%       -exp(d) * 1/H * 8  * 1/a^2 * w_d * (-v   + F(x,y))];





% assume(p > 0.5)
%% Perfect Newton-based Source Seeker:

% 
% b0 = [0;0;h*(-v   + F(x,y));1-e1*d  ;-w_d*d;(-e2 + (-v + F(x,y))) ];
% 
% b1 =   b * e1 * (-v   + F(x,y)) *  [cos(theta);...
%                                    sin(theta);...
%                                             0;...
%                                             0;0;0];
% 
% b2 = a *e2 *[cos(theta);...
%           sin(theta);...
%                    0;...
%                    0;0;0];
%                                                                                                          
% b3 = [ 0;...
%        0;...
%        0;...
%        0;...
%        8 * 1/(a)^2 * w_d *  (-v   + F(x,y));0];
%% Inputs

p = 1;

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

var = [x,y,v,d];
% var = [x,y,v,e1,d,e2];
% B = [b1,b2];
% U = [u1,u2];
% var = [x,y,v];
% [LBS,LBSlim] = secondOrdLBS(B,U,var)

[LBS,LBSlim] = secondOrdLBS(B,U,var,k);

LieBracketSys = b0 + LBSlim

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
% NominalSys = b0 + simplify(gg);
% 
% %% 
% aa = LBS(1);
% bb = LBS(2); 
% AA = children(expand(aa));
% GG = children(expand(bb));
% 
% for i = 1:length(AA)
% 
% BB(i,:) = AA(i);
% 
% end
% 
% for i = 1:length(GG)
% 
% CC(i,:) = GG(i);
% 
% end
% 
%     
% % BB
% % CC
% %     
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
 
 