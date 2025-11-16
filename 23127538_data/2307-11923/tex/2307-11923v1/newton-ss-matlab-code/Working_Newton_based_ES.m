%% A Simulation of the Perfect Source Seeker

clc;
clear;
scalesize = 0.6;
latex_fs = 12;
fs = latex_fs/scalesize;
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontSize', fs)
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontSize', fs)
set(0,'DefaultAxesFontName','CMU Serif')



%% Parameter Initialization

Tsim = 500;


H  = 3* [1, 0; 0, 1];

% F = @(x1,x2)  -1/2 * [x1; x2].' .* H .* [x1; x2];% static map

F = @(x1, x2) 1/2 * 3 * (x1.^2 + x2.^2) ;

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);


%% Parameters
w = 500; 
k1 = 1;
k2 = 1.2; 
p_star = 0.51;
ro = 0.012;
w_d = 0.4;
w_y = 20;
w_z = 0.5;

k_z1 = k1 + k1;
k_z2 = k1 + k2;
k_z3 = k2 + k2;

a_z1 = 8*k1.^2;
a_z2 = 4*k1*k2;
a_z3 = 8*k2.^2;

p_star2 = 1-p_star;

p_star3 = 2-2*p_star;


%% Newton based source seeking;


g = @(t,x) [ro .* x(3) + w.^(p_star) .* sin(k1 .* w .* t);...
            ro .* x(4) + w.^(p_star) .* sin(k2 .* w .* t);...
            - w_d .* ( x(5)  + x(7) .* x(3) + x(8) .* x(4) );...
            - w_d .* ( x(6)  + x(8) .* x(3) + x(9) .* x(4) );...
            - w_y .* (x(5) + 2 .* w.^(p_star2) .* k1 .* F(x(1), x(2)) .* cos(k1 .* w .* t) );... 
            - w_y .* (x(6) + 2 .* w.^(p_star2) .* k2 .* F(x(1), x(2)) .* cos(k2 .* w .* t) );...
            - w_z .* (x(7) - a_z1 .* w.^(p_star3) .* F(x(1), x(2)) .* cos(k_z1 .* w .* t) );...
            - w_z .* (x(8) - a_z2 .* w.^(p_star3) .* F(x(1), x(2)) .* cos(k_z2 .* w .* t) );...
            - w_z .* (x(9) - a_z3 .* w.^(p_star3) .* F(x(1), x(2)) .* cos(k_z3 .* w .* t) )];


        
        
%% Initial Conditions

x0 = [2;2];

d0 = zeros(2,1);
y0 = zeros(2,1);

z0 = [1;0;1];

xo = [x0;d0;y0;z0];



tspan = [0,Tsim]; 


%% 
sol = ode45(g, tspan, xo);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
d1 = sol.y(3,:);
d2 = sol.y(4,:);
y1 = sol.y(5,:);
y2 = sol.y(6,:);
z11 = sol.y(7,:);
z12 = sol.y(8,:);
z22 = sol.y(9,:);



plot(x1,x2);





