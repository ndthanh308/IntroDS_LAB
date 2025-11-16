%% Simulation
% NominalSys = q*w^(7/10)*cos(k*t*w)*cos(t*w0) + d*w^(3/10)*sin(k*t*w)*cos(t*w0)*(F(x, y) - h*z_e)
%              q*w^(7/10)*cos(k*t*w)*sin(t*w0) + d*w^(3/10)*sin(k*t*w)*sin(t*w0)*(F(x, y) - h*z_e)
%                                                                                 F(x, y) - h*z_e
%                                                                                w_d*(- z*d^2 + d)
%                                             (8*k^2*w^(3/5)*w_z*cos(2*k*t*w)*F(x, y))/q^2 - w_z*z


clc;
clear;
scalesize = 0.75;
latex_fs = 12;
fs = latex_fs/scalesize;
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontSize', fs)
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontSize', fs)
set(0,'DefaultAxesFontName','CMU Serif')

%% Parameters
p2 = 0.51;  % p2 > 0.5

p1 = 1-p2;

p3 = 2 - 2*p2;

w = 80;
w0 = 4;
q = 1;
c = 1;
k = 1;
h = 1;
w_d = 1;
w_z = 3;
w_l = 0.4;
mu = 0.001;
q_hess = 1/10;
x_star = 1;
y_star = -1;
F_star = 1;
F = @(x,y) F_star - 1/2 * q_hess * ((x-x_star).^2 + (y-y_star).^2); 

%% System
for i = 1:2
    
if i == 1

f = @(t,x) [ q*w^(p2)*cos(k*t*w)*cos(t*w0) + c*w^(p1)*sin(k*t*w)*cos(t*w0)*(F(x(1),x(2)) - h*x(3));...
             q*w^(p2)*cos(k*t*w)*sin(t*w0) + c*w^(p1)*sin(k*t*w)*sin(t*w0)*(F(x(1),x(2)) - h*x(3));...
                                                                                F(x(1),x(2)) - h*x(3) ];
                                        
x0 = [4;-4;0];    

end
if i == 2    
f = @(t,x) [ q*w^(p2)*cos(k*t*w)*cos(t*w0) +  c*x(4)*w^(p1)*sin(k*t*w)*cos(t*w0)*(F(x(1),x(2)) - h*x(3));...
             q*w^(p2)*cos(k*t*w)*sin(t*w0) +  c*x(4)*w^(p1)*sin(k*t*w)*sin(t*w0)*(F(x(1),x(2)) - h*x(3));...
                                                                                  F(x(1),x(2)) - h*x(3);...
                                                                             w_d*(x(4) - x(5)*x(4).^2 );...
                                -w_z* (x(5)  -   8*k^2*w^(p3)*cos(2*k*t*w)*(F(x(1),x(2)) - h*x(3))/q^2 );...
                                                                 w_l * ( F(x(1),x(2)) - h*x(3) - x(6) )];
                                        
x0 = [4;-4;0;2;0.4;10];                                        
end                                        
Tsim = 30;





tspan = [0,Tsim]; 

 
sol = ode45(f, tspan, x0);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x3 = sol.y(3,:);
x4 = zeros(size(x1));
x5 = x4;
if i == 2
x4 = sol.y(4,:);
x5 = sol.y(5,:);
x6 = sol.y(6,:);
end
% x4 = sol.y(4,:);


%% Plots
figure(1);

subplot(1,2,1);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(x1,x2,'LineWidth',2);
% plot(t,x_star + 0.*t,'-.','LineWidth',2);
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
axis tight;
grid on;
if i == 2
plot(x_star,y_star,'x','LineWidth',2,'MarkerSize',8)
legend('Nominal Seeker','Newton Seeker','Source', 'Interpreter','latex');
end

subplot(1,2,2);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,F(x1,x2),'LineWidth',2);
% 
xlabel('$t$','Interpreter','latex')
ylabel('$F(x)$','Interpreter','latex')
% axis equal;
grid on;

legend('Nominal Seeker','Newton Seeker','Source', 'Interpreter','latex');
if i == 2
plot(t,F_star + 0.*t,'-.','LineWidth',2);
end


if i == 2
figure(2);
set(gca,'TickLabelInterpreter', 'latex')
subplot(1,2,1);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x4,'LineWidth',2);
plot(t,1/q_hess + 0.*t,'-.','LineWidth',2);
xlabel('$t$','Interpreter','latex')
ylabel('$d$','Interpreter','latex')
grid on;
subplot(1,2,2);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x5,'LineWidth',2);
plot(t,q_hess + 0.*t,'-.','LineWidth',2);
ylabel('$z$','Interpreter','latex')
xlabel('$t$','Interpreter','latex')
grid on;
end

figure(3);
subplot(1,2,1);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x1,'LineWidth',2);
% plot(t,x_star + 0.*t,'-.','LineWidth',2);
xlabel('$t$','Interpreter','latex')
ylabel('$x_1$','Interpreter','latex')
grid on;
legend('Nominal Seeker','Newton Seeker', 'Interpreter','latex');

if i == 2
plot(t, x_star + 0.*t,'-.','LineWidth',2);
end

subplot(1,2,2);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x2,'LineWidth',2);
% 
xlabel('$t$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
grid on;

legend('Nominal Seeker','Newton Seeker', 'Interpreter','latex');


if i == 2
plot(t, y_star + 0.*t,'-.','LineWidth',2);
end

end

                                                                              