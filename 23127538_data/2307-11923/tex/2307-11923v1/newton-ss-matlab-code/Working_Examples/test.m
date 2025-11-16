%% 
clear; 
clc;
syms x1 x2 w0 t d z H c k k_j x1_star x2_star w wd wz s a tau p;

A = -a/(2) * [cos(w0*t)^2, cos(w0*t)*sin(w0*t);...
              cos(w0*t)*sin(w0*t), sin(w0*t)^2];
 
Z = [sin(w0 * t), cos(w0 * t);...
    -cos(w0 * t), sin(w0 * t)];

P = sym('P', [2 2]);
P(2,1) = P(1,2);
J = [0 w0;-w0, -a/(2)];
Jtilde = [0 w0;-w0, -a*H/(2)];
Aprime  = Jtilde  - 1/H * eye(2);
J0 = [0 w0;-w0, 0];
L = [0 0;0, -a/(2)];
Q = sym(eye(2));
N = sym(zeros(2));
B = J.'*P + P*J + Q;
eqns = B(:) == N(:);
for i = 1:length(eqns)

eqns1(:,i) = eqns(i);
end
eqns2 = [eqns1(1),eqns1(2),eqns1(4)];

F = solve(eqns2,[P(1,1),P(1,2),P(2,2)]);

P0 = [F.P1_1,F.P1_2;F.P1_2,F.P2_2]

% Pt = Z * P0 * Z.';
% Ptdot = Z * (J0.' * P0 + P0 * J0) * Z.'; 
% 
% U = -(J0.' * P0 + P0 * J0) - eye(2);
% 
% G = P0 * L + L.' * P0;
% assume(H > 0);
% assume(c > 0);
% assume(k > 0);
% assume(w0 > 0);


Vdot = (-z.'*z + d*z.' * G * z - d^2 - d^2 * z.' * P0 *z)/(1 + z'*P0*z);
%%
syms x y;


xdot = -x -x*y;
ydot = -y;
syms v1(x) v2(y) v(x,y);

clc;

V = log(1 + x^2) + y^2 ;

% % V =  (x^2)/(1+x^2);
% %+  1/2 * y ^2  + 1/2 * log( 1 + x^2 ) ;
% 
Vdot = diff(V,x)*xdot + diff(V,y) * ydot;
Vdot = expand(Vdot);
simplify(Vdot)

% A = subs(A,w0,1);
%  
% syms p1(t) p2(t) p3(t) p4(t) q1(t) q2(t) q3(t) q4(t);
% 
% P = [p1(t),p2(t);p2(t),p4(t)];
% 
% Q = [q1(t),q2(t);q2(t),q4(t)];
% 
% Pdot = -P*A - A.' * P - d*eye(2);

% Pdot= simplify(expand(Pdot));
% syms x1(t) x2(t) a;
% 
% X = [x1(t);x2(t)];
% ode1 = diff(x1(t),t) ==  -x1(t)*cos(t)^2 - sin(t)*x2(t)*cos(t);
% ode2 = diff(x2(t),t) ==  -x2(t)*sin(t)^2 - sin(t)*x1(t)*cos(t);
% 
% cond1 = x1(0) == 1;
% cond2 = x2(0) == 0;
% 
% cond = [cond1;cond2];
% ode = [ode1;ode2];
% 
% ySol = dsolve(ode,cond)
% 
% f1 = @(a) - a/2 - ((a - 2).*(a + 2)).^(1/2)./2;
% f2 = @(a)   ((a - 2).*(a + 2)).^(1/2)./2 - a/2;
% xdot = -c*Hp/(2*k) * d * A * x;
% 
% ddot = wd * (1 - z*d);
% 
% zdot = -wz * (z-Hp);
% 
% xxdot = [xdot;ddot;zdot];
% 
% V = 1/2 * x.'*x + 1/2 * (d-1/z)^2  + 1/2 * (z-Hp)^2;
% 
% gradV = gradient(V,[x1,x2,d,z]);
% 
% 
% C = [wd * z, z^(-2)/2;z^(-2)/2, wz]






%%
% Vdot = x' * xdot + 1/wd * (d - Hp^(-1)) * ddot + 1/wz * (z-Hp) * zdot;
% assume(a,'rational')
% assume(a,'positive')
% assume(k,'rational')
% assume(k,'positive')
% assume(a ~= 1);
% a = 1.9;
% ui = sin(k*w*p);
% f = cos(a*k * w*s)*ui - subs(ui,p,s) * cos(a* k * w * p);
% fu = int(f,p,0,s);
% 
% intfu = int(fu,s,0,tau)


% intfu2 = (a*(2*cos(k*tau*w)*cos(a*k*tau*w) - 2) + sin(k*tau*w)*sin(a*k*tau*w) + a^2*sin(k*tau*w)*sin(a*k*tau*w))/(a*k^2*w^2*(a^2 - 1));

% intfu2 = expand(intfu2);

% expr = ui * intfu2;

% expr = subs(ui,p,tau) * intfu;
% expr = expand(expr);
% 
% T = 2 * pi/(k*w);
% %  0.1262
% aaa = 1/(3*T) * int(expr,tau,0,T);
% 
% double(k^2*w^2 * aaa)  