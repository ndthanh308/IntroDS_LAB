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

Tsim = 6;

F = @(r) -1/2  * r.^2;

k = 1.5;
w = 20;

x0 = [4,-1];

tspan = [0, Tsim];
opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

g = @(t,x) sqrt(w) * [cos(w*t - k*F(x(1)) - x(2));... 
                       1/x(1) .* sin(w*t - k*F(x(1)) - x(2)) ];

sol = ode45(g, tspan, x0);

t = sol.x;

r = sol.y(1,:);
gamma = sol.y(2,:);
%% Plots

figure(1);
hold on;
plot(t,r,'LineWidth',2)
plot(t,gamma,'LineWidth',2)
axis tight;
grid on;
legend('\rho(t)','\gamma(t)')
hold off;

% figure(2);
% plot(r.*cos(gamma), r.*sin(gamma),'LineWidth',2)