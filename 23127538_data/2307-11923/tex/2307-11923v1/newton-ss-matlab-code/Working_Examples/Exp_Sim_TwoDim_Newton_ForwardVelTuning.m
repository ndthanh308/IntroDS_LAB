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
p = 0.51;  % p2 > 0.5
% p = 0.51;
p = 1;
p1 = 1-p;

p3 = 2 - 2*p;

w = 50;
w0 = 1;
q = 1;
% c = 1; % Parameter value for the Riccati estimator
c = 0.2;% Parameter value for the exponential estimator
k = 1;
h = 1;
w_d = 0.05;
w_l = 0.5;
w_z = 0.5; % Parameter value for the Riccati estimator
% w_z = 0.45; % Parameter value for the exponential estimator
q_hess = 1/10;
x_star = 1;
y_star = -1;
F_star = 1;
F = @(x,y) F_star - 1/2 * q_hess * ((x-x_star).^2 + (y-y_star).^2); 

Aavg = @(t) [cos(w0*t).^2, cos(w0*t).*sin(w0*t);...
             cos(w0*t).*sin(w0*t), sin(w0*t).^2];
%% System
for i = 1:2
    
if i == 1

f = @(t,x) [ c*w^(p)*cos(k*t*w)*cos(t*w0) + q*w^(1-p)*sin(k*t*w)*cos(t*w0)*(F(x(1),x(2)) - x(3));...
             c*w^(p)*cos(k*t*w)*sin(t*w0) + q*w^(1-p)*sin(k*t*w)*sin(t*w0)*(F(x(1),x(2)) - x(3));...
                                                                        h*(F(x(1),x(2)) - x(3)) ];
                                        
x0 = [4;-4;0];    

end
if i == 2    
%% System that estimates the Hessian and the inverse separately     
% f = @(t,x) [ q*w^(p2)*cos(k*t*w)*cos(t*w0) + c*x(4)*w^(p1)*sin(k*t*w)*cos(t*w0)*(F(x(1),x(2)) - h*x(3));...
%              q*w^(p2)*cos(k*t*w)*sin(t*w0) + c*x(4)*w^(p1)*sin(k*t*w)*sin(t*w0)*(F(x(1),x(2)) - h*x(3));...
%                                                                                  F(x(1),x(2)) - h*x(3);...
%                                                                            w_d*(1 - x(4)*x(5));...
%                            -w_z* (x(5)  -   8*k^2*w^(p3)*cos(2*k*t*w)*(F(x(1),x(2)) - h*x(3))/q^2 ) ];

%% System that estimates the inverse of the Hessian directly
%% Ricatti Estimator
% original version
% f = @(t,x) [ c*w^(p)*cos(k*t*w)*cos(t*w0) + x(4)*w^(1-p)*sin(k*t*w)*cos(t*w0)*(F(x(1),x(2)) - x(3));...
%              c*w^(p)*cos(k*t*w)*sin(t*w0) + x(4)*w^(1-p)*sin(k*t*w)*sin(t*w0)*(F(x(1),x(2)) - x(3));...
%                                                                              h*(F(x(1),x(2)) -  x(3));...
%                       w_z* x(4)*(1  -   x(4) * 8*k^2*w^(2-2*p)*cos(2*k*t*w)*(F(x(1),x(2)) - x(3))/c^2) ];

% low pass hessian
f = @(t,x) [  c*w^(p)*cos(k*t*w)*cos(t*w0) +  x(4)*w^(1-p)*sin(k*t*w)*cos(t*w0)*h*(F(x(1),x(2)) -  x(3));...
              c*w^(p)*cos(k*t*w)*sin(t*w0) +  x(4)*w^(1-p)*sin(k*t*w)*sin(t*w0)*h*(F(x(1),x(2)) -  x(3));...
                                                           h*(F(x(1),x(2)) -  x(3));...
                      w_z * x(4)*(1  -   x(4) * x(5) );...
                      w_l * ( 8*k^2*w^(2-2*p)*cos(2*k*t*w)*(F(x(1),x(2)) -  x(3))/c^2 - x(5))];
                  
 
% low pass d

% f = @(t,x) [  c*w^(p)*cos(k*t*w)*cos(t*w0) +  x(5)*w^(1-p)*sin(k*t*w)*cos(t*w0)*h*(F(x(1),x(2)) -  x(3));...
%               c*w^(p)*cos(k*t*w)*sin(t*w0) +  x(5)*w^(1-p)*sin(k*t*w)*sin(t*w0)*h*(F(x(1),x(2)) -  x(3));...
%                                                            h*(F(x(1),x(2)) -  x(3));...
%               w_z * x(4)*(1  -   x(4) * 8*k^2*w^(2-2*p)*cos(2*k*t*w)*(F(x(1),x(2)) -  x(3))/c^2 );...
%               w_l * (x(4) - x(5))];
     

% f_avg = @(t,x) [-x(4) * c*q_hess/(2*k) * Aavg(t) * [x(1)-x_star;...
%                                                     x(2)-y_star];... 
%                                                     h*(F(x(1),x(2)) - x(3));...
%                                                     w_z * x(4)*(1-q_hess*x(4))];
%                                                 

x0 = [4;-4;0;1;1]; % Low-pass filter
% x0 = [4;-4;0;1]; % Ricatti Estimator
%% Exponential Estimator
% f = @(t,x) [ c*w^(p)*cos(k*t*w)*cos(t*w0) + q*exp(x(4))*w^(1-p)*sin(k*t*w)*cos(t*w0)*(F(x(1),x(2)) - x(3));...
%              c*w^(p)*cos(k*t*w)*sin(t*w0) + q*exp(x(4))*w^(1-p)*sin(k*t*w)*sin(t*w0)*(F(x(1),x(2)) - x(3));...
%                                                                              h*(F(x(1),x(2)) -  x(3));...
%                       w_z* (1  -   exp(x(4)) * 8*k^2*w^(2-2*p)*cos(2*k*t*w)*(F(x(1),x(2)) - x(3))/c^2) ];
%                   
% f_avg = @(t,x) [-exp(x(4)) * c*q_hess/(2*k) * Aavg(t) * [x(1)-x_star;...
%                                                     x(2)-y_star];... 
%                                                     h*(F(x(1),x(2)) - x(3));...
%                                                     w_z *(1-q_hess*exp(x(4)))];                  
%                   

% x0 = [4;-4;0;1];   % Exponential estimator

end                                        
Tsim = 100;
tspan = [0,Tsim]; 

sol = ode45(f, tspan, x0);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x4 = zeros(size(x1));
%% 
if i == 2
x3 = sol.y(3,:);
x4 = sol.y(4,:);
x5 = sol.y(5,:);
% sol_avg = ode45(f_avg,tspan,x0);
% 
% tavg = sol_avg.x;
% x1avg = sol_avg.y(1,:);
% x2avg = sol_avg.y(2,:);
% davg = sol_avg.y(4,:);
end



%% Plots
figure(1);
title('$p = 1$','Interpreter', 'latex')

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
title('Low-pass filtered $\cos(2k\omega t) (F(x) - \nu)$','Interpreter', 'latex')
set(gca,'TickLabelInterpreter', 'latex')
% subplot(1,2,1);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x4,'LineWidth',2);
% plot(t,exp(x4),'LineWidth',2);
plot(t,1/q_hess + 0.*t,'-.','LineWidth',2);
xlabel('$t$','Interpreter','latex')
ylabel('$d(t)$','Interpreter','latex')
% ylabel('$e^{d(t)}$','Interpreter','latex')
grid on;

end





%%
% figure(3);
% subplot(1,2,1);
% set(gca,'TickLabelInterpreter', 'latex')
% hold on;
% plot(t,x1,'LineWidth',2);
% % plot(t,x_star + 0.*t,'-.','LineWidth',2);
% xlabel('$t$','Interpreter','latex')
% ylabel('$x_1$','Interpreter','latex')
% grid on;
% legend('Nominal Seeker','Newton Seeker', 'Interpreter','latex');
% 
% if i == 2
% plot(t, x_star + 0.*t,'-.','LineWidth',2);
% end
% 
% subplot(1,2,2);
% set(gca,'TickLabelInterpreter', 'latex')
% hold on;
% plot(t,x2,'LineWidth',2);
% % 
% xlabel('$t$','Interpreter','latex')
% ylabel('$x_2$','Interpreter','latex')
% grid on;
% 
% legend('Nominal Seeker','Newton Seeker', 'Interpreter','latex');


% if i == 2
% plot(t, y_star + 0.*t,'-.','LineWidth',2);
% end
%% 
% if i == 2
% figure(4);
% % subplot(1,2,1);
% hold on;
% plot3(x1, x2, exp(x4),'LineWidth',1,'Color',[0,0,0]);
% plot3(x1avg,x2avg,exp(davg),'-.','LineWidth',2,'Color','r');
% plot3(x_star,y_star,1/q_hess,'x','LineWidth',5,'MarkerSize',10)
% legend('Newton Seeker (2)','Average Newton Seeker (3)','Source + 1/H', 'Interpreter','latex');
% grid on;
% xlabel('$\bar{x}_1$','Interpreter','latex')
% ylabel('$\bar{x}_2$','Interpreter','latex')
% zlabel('$\bar{d}$','Interpreter','latex')
% figure(5);
% grid on;
% hold on;
% plot(x1,x2,'LineWidth',1,'Color',[0,0,0])
% plot(x1avg,x2avg,'-.','LineWidth',2,'Color','r')
% plot(x_star,y_star,'x','LineWidth',2,'MarkerSize',8)
% legend('Newton Seeker (2)','Average Newton Seeker (3)','Source', 'Interpreter','latex');
% xlabel('$\bar{x}_1$','Interpreter','latex')
% ylabel('$\bar{x}_2$','Interpreter','latex')
% axis equal
% end
end

                                                                              