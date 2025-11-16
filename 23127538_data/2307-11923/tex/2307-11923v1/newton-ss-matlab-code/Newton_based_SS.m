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


% F = @(x1,x2)  -1/2 * [x1; x2].' .* H .* [x1; x2];% static map

h  = 4;
H  = h* [1, 0; 0, 1];
F = @(x1)  -1/2 * h * (x1.^2) ;

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);


%% Parameters
w = 200; 
k1 = 2;
k2 = 1; 
p_star = 0.5;
ro = 0.2;

w_d = 10;

w_y = 5;

w_z = 10;

k_z1 = k1 + k1;
k_z2 = k1 + k2;
k_z3 = k2 + k2;

a_z1 = 8*k1.^2;
a_z2 = 4*k1*k2;
a_z3 = 8*k2.^2;

p_star2 = 1-p_star;

p_star3 = 2-2*p_star;


%% One Dimensional Newton based Extremum Seeking;

% g = @(t,x) [ro .* x(2) + w.^(p_star) .* sin(k1 .* w .* t);...
%             - w_d .* (x(3)  + x(4) .* x(2) );...
%             - w_y .* (x(3) + 2 .* w.^(p_star2) .* k1 .* F(x(1)) .* cos(k1 .* w .* t) );... 
%             - w_z .* (x(4) - a_z1 .* w.^(p_star3) .* F(x(1)) .* cos(k_z1 .* w .* t) )];


%% One Dimensional Newton based SOURCE SEEKING;        
    w = 500;
    k1 = 1;
    k2 = 1.5;
    w_d = 1;
    w_z = 5;
    p_star = 0.52;
    
    k_z1 = k1 + k1;
    a_z1 = 8*k1.^2;
    p_star3 = 2-2*p_star;
    
    
    g = @(t,x) [ w.^(p_star) .* cos( w .* t - k2*F(x(1))) + w.^(p_star) * sin(k1*w*t);...
              - w_d .* (-x(2)  + x(3) .* x(2).^2 );...
              - w_z .* (x(3) - a_z1 .* w.^(p_star3) .* F(x(1)) .* cos(k_z1 .* w .* t) )];


%% Initial Conditions

Tsim = 5;

d0 = [1;0;0;1];

z0 = [1;0;1];

% xo = [-2;-2;d0;z0];

xo = [-3;1;1];

tspan = [0,Tsim]; 


%% 
sol = ode45(g, tspan, xo);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x3 = sol.y(3,:);
% x4 = sol.y(4,:);
% y2 = sol.y(6,:);
% z11 = sol.y(7,:);
% z12 = sol.y(8,:);
% z22 = sol.y(9,:);

%% Plots
figure(1);

subplot(1,3,1);
hold on
plot(t,x1);
plot(t,0.*t,'-.');
% yline(0,'-.','LineWidth',1.2);
ylabel('x')
grid on; 

subplot(1,3,2);
hold on;
plot(t,x3,'LineWidth',2)
plot(t,0.*t + h,'-.');
% yline(h,'-.','LineWidth',1.2)
legend('z')
grid on;

subplot(1,3,3);
hold on;
plot(t,x2,'LineWidth',2)
plot(t,0.*t + 1/h,'-.');
% yline(1/h,'-.','LineWidth',1.2)
% yline(0,'-.','LineWidth',1)
legend('d')
grid on;




