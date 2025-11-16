%% Simulation

clc;
clear;

w = 200;
k = 0.5;


Tsim = 50;

w_z = 3;
w_y = 2;
h = 1/4;
x_star = 3;
F = @(x)    - 1/2 *  h * ((x-x_star).^2 ) ;

%%
% NominalSys =   ga^(1/2)*w^(3/4)*cos(t*w^(1/2) - (y*F(x))/w)
%                w_y*(- z*y^2 + y)
%           - w_z*z - (8*w^(1/2)*w_z*cos(2*t*w^(1/2))*cos(F(x)/w)*sin(F(x)/w))/ga

g= @(t,x) [(k)^(1/2) * w^(3/4) * cos(t*w^(1/2) - (x(2) * F(x(1))) /w);
           w_y * (x(2) - x(3) * x(2).^2);
           -w_z * ( x(3) - (8*w^(1/2)*cos(2*t*w^(1/2))*cos(F(x(1))/w)*sin(F(x(1))/w))/k )];


x0 = [-3;0.1;0.3];

tspan = [0,Tsim]; 

 
sol = ode45(g, tspan, x0);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x3 = sol.y(3,:);
% x4 = sol.y(4,:);


%% Plots
figure(1);

subplot(1,3,1);
hold on;
plot(t,x1,'LineWidth',2);
plot(t,x_star + 0.*t,'-.','LineWidth',2);
xlabel('t')
ylabel('x_1')
grid on;


subplot(1,3,2);
hold on;
plot(t,x2,'LineWidth',2);
plot(t,1/h + 0.*t,'-.','LineWidth',2);
ylabel('1/h')
xlabel('t')
grid on;


subplot(1,3,3);
hold on;
plot(t,x3,'LineWidth',2);
plot(t,h + 0.*t,'-.','LineWidth',2);
ylabel('h')
xlabel('t')
grid on;

