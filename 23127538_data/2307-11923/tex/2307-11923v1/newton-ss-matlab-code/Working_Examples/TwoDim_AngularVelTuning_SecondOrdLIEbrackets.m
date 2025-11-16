%% Second Order Lie Bracket Averaging
% just enter the vector fields and the periodic inputs and it will spit out
% the second order Lie bracket averaged system, son!

clc;
clear;

syms x y z_e d z F(x,y)  h k w w0 t p w_d w_z p0 p1 p2 p3 p4 p5 c1 c2 ga ga2 q q_hess;


assume([x y d z F(x,y) h k w w0 t p w_d w_z p1 p2 p3 p4 p5 c1 c2 ga ga2 q q_hess],'real')


F(x,y) =  -1/2 * q_hess*(x^2 + y^2);





k = 1;
b0 = [0;0; w_d*d];


theta = h * F(x,y);


b1 = [cos(theta);...
      sin(theta);...
               0];

b2 =  [sin(theta);...
      -cos(theta);...
                0];
  

b3 = [ 0;...
       0;...
       -8 *   w_d * d^2 * F(x,y)];
    

p = 0.51;
u1 = w^p * cos(w * t);
u2 = w^p * sin(w * t);


p3 = 2 - 2*p;

u3 = w^p3 * cos(2 *  w * t);

% u4 = w^p2 * sin(2 * k * w *t);
% u4 = w^(2-2*p) * sin(  k * w * t);



%% 
B = [b1,b2,b3];

U = [u1,u2,u3];

var = [x,y,d];


% [LBS,LBSlim] = secondOrdLBS(B,U,var)

[LBS,LBSlim,~] = secondOrdLBS(B,U,var,k);

LieBracketSys = b0 + LBSlim
% 
aa = sum(B(1,:).*U); 
bb = sum(B(2,:).*U);
cc = sum(B(3,:).*U);

% dd = sum(B(4,:).*U);
% ee = sum(B(5,:).*U);
%%

gg = [aa;bb;cc];

NominalSys = b0 + simplify(gg);

%% 
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
% cc = LBS(4);
% 
% GD = children(expand(cc));
% for i = 1:length(GD)
% 
% DD(i,:) =  GD(i);
% end
% 
% % DD
% %    
% % sum( limit(DD,w,inf))
% 
% 