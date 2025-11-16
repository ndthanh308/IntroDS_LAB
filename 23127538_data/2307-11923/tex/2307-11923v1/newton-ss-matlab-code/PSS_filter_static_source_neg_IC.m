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


% s = @(t) (1 + tanh(5*cos(t./20))).*(t<40) + (t>=40).*0;
% s = @(t) (1 + tanh(5*cos((t-25)./80)));
% y = @(x, y,t)  - 1/2 * ((x  + s(t) ).^2 + (y ).^2);

%% Parameter Initialization

y = @(x,y,t)  - 1/2 * (x.^2 + (y-1).^2); % static map

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

mu = 0.001;
w = 15;
k = 1.2;
wl = 0.2; 

S = sqrt(w); % Saturation Level 
sat = @(x) min(S, max(-S, x)); % Nonsmooth Saturation
sat = @(x) sqrt(w)*2/(1 + exp(-2*x)) - sqrt(w); % Smooth Saturation

%% Simulation: Nominal Seeker vs Perfect Seeker

for i = 1:2
    
    if i == 1
        % Nominal Seekers forward velocity
        kt = @(x5,t) 1;
%         kt = @(x5,t) sqrt(w);
    else  
        % Perfect Seeker  forward velocity
        
        eps = 0.001;
        c = 1000;  % great results for x50 = -2;
        c = 50000; % great results for x50 = -1;
       kt = @(x5,t) (2/(2*eps + pi))*(atan(c*x5.^4) + eps);

    end


fp = @(t,x)     [ sqrt(w) *kt(x(5),t)  * cos(x(3));...
                  sqrt(w) *kt(x(5),t)  * sin(x(3));...
                  (w -   k/mu * (y(x(1),x(2),t) - x(4)));
                   1/mu *(y(x(1),x(2),t) - x(4));
                   wl*( 1/mu*(y(x(1),x(2),t) - x(4)) - x(5) )];
                             
                      
                              

tspan = [0,30]; 


x05 = -2; % great results for  c = 1000
x05 = -1; % great results for c = 50000
x0 = [-2;2;0;y(-2,2,0);x05];
sol = ode15s(fp, tspan, x0, opts);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x3 = sol.y(3,:);
x4 = sol.y(4,:);
x5 = sol.y(5,:);
zdot =  1/mu *(y(x1,x2,t) - x4);


%% Plots 

%% Figure 1 represents the trajectories of both Seekers (x1 vs x2 plot)
figure(1)
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(x1,x2,'LineWidth',i);
grid on;
if i == 2
plot(0,1,'x','MarkerSize',10,'LineWidth',2,'color',[0,0,0])
end
xlabel( '$x_1$','Interpreter','latex','FontSize', fs)
ylabel('$x_2$','Interpreter','latex', 'Fontsize',fs)
legend('Nominal seeker','Perfect seeker','Source $x^* = [0,1]^\top$','Interpreter','latex');
axis equal;
hold off;

%% Figure 2 represents the trajectoris of both Seekers vs time (x1 vs t plot)

% figure(2)
% set(gca,'TickLabelInterpreter', 'latex')
% hold on;
% plot(t,x1,'LineWidth',2);
% plot(t,x2,'LineWidth',2);
% yline(0,'-.','LineWidth',1.5)
% yline(1,'-.','LineWidth',1.5)
% grid on;
% axis equal;
% xlabel( '$x_1$','Interpreter','latex','FontSize', fs)
% ylabel('$x_2$','Interpreter','latex', 'Fontsize',fs)
% % legend('Nominal seeker','Perfect seeker','Interpreter','latex');
% hold off;



%% Figure 2 represents the forward velocity

u1 = sqrt(w)*kt(x5,t);

if i == 2
figure(3);
subplot(1,3,1)
set(gca,'TickLabelInterpreter', 'latex')
plot(t,u1,'LineWidth',2);
yline(sqrt(w),'-.','LineWidth',2)
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$u_1(t) = \sqrt{\omega} \alpha(\eta(t))$','Interpreter','latex', 'Fontsize',fs)
%
subplot(1,3,2)
set(gca,'TickLabelInterpreter', 'latex')
plot(t,x5,'LineWidth',1);
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$\eta(t)$','Interpreter','latex', 'Fontsize',fs)

subplot(1,3,3);
tt = linspace(-1,0.5);
set(gca,'TickLabelInterpreter', 'latex')
plot(tt,sqrt(w)*kt(tt,0),'LineWidth',2);
yline(sqrt(w),'-.','LineWidth',1.5);
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
% ylabel('$\frac{1}{\mu}(y-z)$','Interpreter','latex', 'Fontsize',fs)
end
%% Figure 3 represents eta





%% Figure 4 represent the field F(x1,x2)
% 
% figure(4);
% set(gca,'TickLabelInterpreter', 'latex')
% hold on;
% plot(t,y(x1,x2,t),'LineWidth',i+1);
% yline(0, '-.','LineWidth',1)
% grid on;
% axis equal;
% ylabel( '$F(x) = -\frac{1}{2}(x_1^2 + x_2^2)$','Interpreter','latex','FontSize', fs)
% xlabel('$t$','Interpreter','latex', 'Fontsize',fs)
% % legend('nominal seeker','perfect seeker');
% set(gca,'TickLabelInterpreter', 'latex')
% ylim([-5,1])


end


%% Distance between end point and source
 
 dist = norm([0,1] - [x1(end),x2(end)])


