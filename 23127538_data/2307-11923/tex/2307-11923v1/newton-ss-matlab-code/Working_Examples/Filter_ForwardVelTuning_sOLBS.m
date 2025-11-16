%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms x y v d z_e alpha w_l F(x,y) F_star h k w w0 t p w_d w_z p c c2 ga ga2 q H x_star y_star;




F(x,y) = F_star -1/2 * H *((x)^2 + (y)^2);

k = 1;

theta = w0 * t;


%% Ricatti Inverse Estimator:


b0 = [0;0;h*(-v   + F(x,y)); w_d*d*(1-d*z_e);-w_l*z_e];

b1 =  d * (-v   + F(x,y)) *  [cos(theta);...
                                   sin(theta);...
                                                 0;...
                                                 0;...
                                                 0];

b2 = alpha *[cos(theta);...
          sin(theta);...
                   0;...
                   0;....
                   0];
                                                                                                         
b3 = [ 0;...
       0;...
       0;...
       0;...
       w_l * 8 * k^2 * 1/alpha^2 *  (-v   + F(x,y))];

  



% assume(p > 0.5)

%% Inputs

p = 1;

u1 =  sin(k* w * t);
u2 = w^(p) * cos(k*w* t);
u3 =  cos(2 * k * w *t) ;

%  
% LieBracketSys =
%  
%  -(H*c*d*cos(t*w0)*(x*cos(t*w0) + y*sin(t*w0)))/(2*k)
%  -(H*c*d*sin(t*w0)*(x*cos(t*w0) + y*sin(t*w0)))/(2*k)
%                   -h*(v - F_star + (H*(x^2 + y^2))/2)
%                                   - H*w_d*d^2 + w_d*d

%% 
B = [b1,b2,b3];

U = [u1,u2,u3];

var = [x,y,v,d,z_e];


% [LBS,LBSlim] = secondOrdLBS(B,U,var)

[LBS,LBSlim] = secondOrdLBS(B,U,var,k);

LieBracketSys = b0 + LBSlim

% 
aa = sum(B(1,:).*U); 
bb = sum(B(2,:).*U);
cc = sum(B(3,:).*U);
% 
dd = sum(B(4,:).*U);
ee = sum(B(5,:).*U);
% 
% 
gg = [aa;bb;cc;;dd;ee];
% 
NominalSys = b0 + simplify(gg);
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
 
 