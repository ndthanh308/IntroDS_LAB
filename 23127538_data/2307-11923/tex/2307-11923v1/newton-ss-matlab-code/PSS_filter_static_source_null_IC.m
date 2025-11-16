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

Tsim = 200;

s  = @(t)  5*1/2 * (2/pi * atan(5*(t-100)) + 1);

y = @(x,y,t)  - 1/2 * (x.^2 + (y-s(t)).^2); % static map

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

mu = 0.001;
w = 10;
k = 1.2;
wl = 0.2; 

%% Saturation Functions, (if needed)
S = sqrt(w); % Saturation Level 
sat = @(x) min(S, max(-S, x)); % Non-smooth Saturation
% sat = @(x) sqrt(w)*2./(1 + exp(-2.*x)) - sqrt(w); % Smooth Saturation

%% Simulation: Nominal Seeker vs Perfect Seeker
xx = linspace(-5,5,500);
for i = 1:2
    
    if i == 1
       % Nominal Seekers forward velocity
        kt = @(x5,t) 1;
       % kt = @(x5,t) sqrt(w);
        K1 = 1;
        K2 = 1;
    else  
% Perfect Seeker  forward velocity
        
       eps = 0.05;
       c = 1;
       kt = @(x5,t) sqrt(eps + c*x5.^4);
       K1 = 10;
       K2 = 1;
    end

u1 = @(x5,t)  sat( sqrt(w).*kt(K2*x5,t) );

fp = @(t,x)     [ u1(x(5),t)  * cos(x(3));...
                  u1(x(5),t)  * sin(x(3));...
                  (w -   k/mu * (y(x(1),x(2),t) - x(4)));
                   1/mu *(y(x(1),x(2),t) - x(4));
                   wl*( K1/mu * (y(x(1),x(2),t) - x(4))   - x(5)  )];
                             
%                     
                              

tspan = [0,Tsim]; 


x05 = 0; 

x0 = [-2;-2;0;y(-2,-2,0);x05];
% 
% sol = ode45(fp, tspan, x0, opts);

sol = ode45(fp, tspan, x0);

t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x3 = sol.y(3,:);
x4 = sol.y(4,:);
x5 = sol.y(5,:);
zdot =  K1/mu *(y(x1,x2,t) - x4);


%% Plots 

%% Figure 1 represents the trajectories of both Seekers (x1 vs x2 plot)
f = figure(1);
subplot(2,3,1)
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(x1,x2,'LineWidth',i);
grid on;
if i == 2
plot(0,0,'x','MarkerSize',5,'LineWidth',1,'color',[0,0,0])
end
xlabel( '$x_1$','Interpreter','latex','FontSize', fs)
ylabel('$x_2$','Interpreter','latex', 'Fontsize',fs)
% legend('Nominal seeker','Perfect seeker','Source $x^* = [s(t), s(t)]^\top$','Interpreter','latex');
% title('s(t) = ', func2str(s),'FontSize',12);
% axis equal;
% hold off;

%% Figure 2 represents the trajectoris of both Seekers vs time (x1 vs t plot)


% figure(2)
subplot(2,3,2);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x1,'LineWidth',i);
if i == 2
% plot(t,s(t),'-.','LineWidth',1.5,'color',[0,0,0],'HandleVisibility','off')
yline(0,'-.','LineWidth',1.5,'color',[0,0,0],'HandleVisibility','off')
end
set(gca,'TickLabelInterpreter', 'latex')
grid on;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$x_1(t)$','Interpreter','latex', 'Fontsize',fs)
% legend('Nominal seeker','Perfect seeker','Interpreter','latex');
set(gca,'TickLabelInterpreter', 'latex')
hold off;

% figure(3);
subplot(2,3,3);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x2,'LineWidth',i);
if i == 2
yline(0,'-.','LineWidth',1.5,'color',[0,0,0],'HandleVisibility','off')
end
set(gca,'TickLabelInterpreter', 'latex')
grid on;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$x_2(t)$','Interpreter','latex', 'Fontsize',fs)
% legend('Nominal seeker','Perfect seeker','Interpreter','latex');
set(gca,'TickLabelInterpreter', 'latex')
hold off;

%% Figure 2 represents the forward velocity
% 

u1 = u1(x5,t);

if i == 2
% figure(3)
subplot(2,3,4);
set(gca,'TickLabelInterpreter', 'latex')
plot(t,u1,'LineWidth',2);
yline(sqrt(w),'-.','LineWidth',2)
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$u_1(t) = \sqrt{\omega} \alpha(\eta(t))$','Interpreter','latex', 'Fontsize',fs)
set(gca,'TickLabelInterpreter', 'latex')
%
subplot(2,3,5);
set(gca,'TickLabelInterpreter', 'latex')
plot(t,K2*x5,'LineWidth',2);
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$\eta(t)$','Interpreter','latex', 'Fontsize',fs)
set(gca,'TickLabelInterpreter', 'latex')
subplot(2,3,6)
set(gca,'TickLabelInterpreter', 'latex')
plot(t,zdot,'LineWidth',2);
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$\frac{1}{\mu}(y-z)$','Interpreter','latex', 'Fontsize',fs)
set(gca,'TickLabelInterpreter', 'latex')
end
%% Figure 5 represents the error between the source and the trajectory of the seeker


% figure(5);    
% err = vecnorm([s(t);s(t)] - [x1;x2]);
% hold on;
% plot(t,err)
% grid on
% xlabel( '$t$','Interpreter','latex','FontSize', fs)
% ylabel('$||x - x^*||$','Interpreter','latex', 'Fontsize',fs)
% legend('Nominal Seeker', 'Perfect Seeker', 'Interpreter','latex')
% hold off;

%% Figure represent the field F(x1,x2)
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
f.WindowState = 'maximized';

%% Distance between end point and source
 
% dist = norm([0,1] - [x1(end),x2(end)])


