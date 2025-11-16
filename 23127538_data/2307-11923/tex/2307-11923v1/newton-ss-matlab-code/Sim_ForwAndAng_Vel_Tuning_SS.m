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

h = 1;

J = @(x,y)    -  h * (x.^2 + y.^2) ;

% opts = odeset('RelTol',1e-6,'AbsTol',1e-9);


%% Parameters
w = 10; 

w0 = 30;

a = 2;
c = 2;

w_z = 0.05;
h1 = 1;

Tsim = 40;


u1 = @(t,x,y,z) c * cos(J(x,y)-z*h1) *cos(w*t) + a * sin(J(x,y)-z*h1) * sin(w*t);
x01 = -2;
y0 =  -2;

k = 0.5;
mu = 0.001;

g = @(t,x) [ sqrt(w0) *  u1(t,x(1),x(2),x(3))    * cos(x(4));...
             sqrt(w0) *  u1(t,x(1),x(2),x(3))    * sin(x(4));...
             -x(3)*h1 + J(x(1),x(2));...
             w0 - k/mu * (J(x(1),x(2)) - x(5));...
             1/mu * ( J(x(1),x(2)) - x(5) )];
        
x0 = [x01;y0;0;0;0];
 
    
% if i == 2    
% g = @(t,x) [ u1(t,x(1),x(2),x(3))  * x(4)  * cos(w0 * t);...
%              u1(t,x(1),x(2),x(3))  * x(4)  * sin(w0 * t);...
%              -x(3)*h1 + J(x(1),x(2));...
%               w_z .* ( x(4) - x(4).^2 * 4/a^2 *  (-x(3)*h1 + J(x(1),x(2)))  .* cos(2 * w * t))];


% x0 = [x01;y0;0;1.1];

tspan = [0,Tsim]; 

 
sol = ode45(g, tspan, x0);
%%
t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);

% x4 = sol.y(4,:);

% x3 = sol.y(3,:);         


%% Plots
figure(1);

% subplot(1,3,1);
hold on;
plot(x1,x2,'LineWidth',2);
xlabel('x_1')
ylabel('x_2')
axis equal;
% legend('i = 1','i = 2')
grid on;


% subplot(1,3,2);
% hold on;
% if i == 2
% plot(t,x4);
% plot(t,1/h + 0.*t,'-.');
% end
% ylabel('1/h')
% xlabel('t')
% grid on;
% 
% subplot(1,3,3);
% hold on;
% plot(t,J(x1,x2),'LineWidth',2)
% % plot(t,0.*t + 1/h,'-.');
% % yline(1/h,'-.','LineWidth',1.2)
% % yline(0,'-.','LineWidth',1)
% xlabel('t')
% ylabel('J(x_1,x_2)')
% legend('i = 1','i = 2')
% grid on;

% end
