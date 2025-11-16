%% Unicycle Source Seeking via Forward Velocity Tuning - Simulations

clc;
clear;
scalesize = 0.6;
latex_fs = 8;
fs = latex_fs/scalesize;
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontSize', fs)
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontSize', fs)
set(0,'DefaultAxesFontName','CMU Serif')

h = 1/2;

J = @(r)   - 1/2 * h * (r.^2) ;

% opts = odeset('RelTol',1e-6,'AbsTol',1e-9);


%% Parameters
w = 10; 

w0 = w/4;

a = 0.1;
c = 20;

w_z = 0.1;
h1 = 1;

Tsim = 100;


u1 = @(t,r,z) c * (J(r)-z*h1) *sin(w*t) + a * w * cos(w*t);


g = @(t,x) [ u1(t,x(1),x(3))   * x(4) * cos(w0 *t - x(2));...
             u1(t,x(1),x(3))   * x(4) * sin(w0 *t - x(2)) / x(1);...
             -x(3)*h1 + J(x(1));...
             w_z .* ( -x(4)   + x(4).^2 * 8/a^2 * ( -x(3)*h1 + J(x(1)) ) .* cos(2*4 * w*t)  )];

% ;... 
%             
x0 = [2;0.5;0;1.9];



tspan = [0,Tsim]; 

 
sol = ode45(g, tspan, x0);
%%
t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x4 = sol.y(4,:);
% x3 = sol.y(3,:);         


%% Plots
figure(1);

subplot(2,2,1);
hold on
plot(t,x1,'LineWidth',2);
plot(t,0.*t,'-.');
% yline(0,'-.','LineWidth',1.2);
xlabel('t')
ylabel('\rho')
% ylim([-0.2,4]);
grid on; 

subplot(2,2,2);
hold on;
plot(t,x2,'LineWidth',2)
% yline(h,'-.','LineWidth',1.2)
xlabel('t')
ylabel('\gamma')
grid on;

subplot(2,2,3);
hold on;
plot(t,x4);
plot(t,1/h + 0.*t,'-.');
ylabel('1/h')
xlabel('t')
grid on;


subplot(2,2,4);
hold on;
plot(x1.*cos(x2),x1.*sin(x2));
xlabel('x_1')
ylabel('x_2')
grid on;

% subplot(1,3,3);
% hold on;
% plot(t,x3,'LineWidth',2)
% plot(t,0.*t + 1/h,'-.');
% % yline(1/h,'-.','LineWidth',1.2)
% % yline(0,'-.','LineWidth',1)
% xlabel('t')
% ylabel('\beta')
% grid on;
