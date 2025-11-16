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
Tsim = 15;
s = [0,0];
y = @(x,y,t)  - 1/2 * ((x-s(1)).^2 + (y-s(2)).^2); % static map

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

mu = 0.0001;
w = 10;
k = 2.5;
wl = 0.4; 

S = sqrt(w); % Saturation Level 
% sat = @(x) min(S, max(-S, x)); % Nonsmooth Saturation
sat = @(x) sqrt(w)*2/(1 + exp(-2*x)) - sqrt(w); % Smooth Saturation

%% Simulation: Nominal Seeker vs Perfect Seeker
dist = zeros(1,2);
for i = 1:2
    
    if i == 1
        % Nominal Seekers forward velocity
        kt = @(x5,t) 1;
%         kt = @(x5,t) sqrt(w);
    else  
        % Perfect Seeker  forward velocity
        
        eps = 0.001;
        c = 5;  
       kt = @(x5,t) (2/(2*eps + pi))*(atan(c*x5.^2) + eps);
%      kt = @(x5,t)  sqrt(eps + x5.^2);
        
%         kt = @(x5,t) 1/(1 + eps) * (1 + eps - exp(-c*x5.^2));
%         kt = @(x5,t)  (eps + x5.^2).^(1/4);
%         u1 = sat(kt(x5,t));
    end


fp = @(t,x)     [ sqrt(w) * kt(x(5),t)  * cos(x(3));...
                  sqrt(w) * kt(x(5),t)  * sin(x(3));...
                  (w -   k/mu * (y(x(1),x(2),t) - x(4)));
                   1/mu *(y(x(1),x(2),t) - x(4));
                   wl*( 1/mu*(y(x(1),x(2),t) - x(4)) - x(5) )];
                             
                      
                              

tspan = [0,Tsim]; 

x05 = 4; % Good results
% x05 = -1;
x0 = [-2;-2;0;y(-2,-2,0);x05];
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
ss = string(s);

source_legend = strcat('Source',' $','x^* = [', ss(1),', ',ss(2),']^\top$');
figure(1)
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(x1,x2,'LineWidth',i);
grid on;
if i == 2
plot(s(1),s(2),'x','MarkerSize',10,'LineWidth',2,'color',[0,0,0])
end
xlabel( '$x_1$','Interpreter','latex','FontSize', fs)
ylabel('$x_2$','Interpreter','latex', 'Fontsize',fs)
legend('Nominal seeker','Perfect seeker',source_legend,'Interpreter','latex');
axis equal;
hold off;

%% Figure 2 represents the trajectoris of both Seekers vs time (x1 vs t plot)

figure(2)
subplot(1,2,1);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x1,'LineWidth',i);
if i == 2
yline(s(1),'-.','LineWidth',1.2,'HandleVisibility','off')
end
grid on;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$x_1(t)$','Interpreter','latex', 'Fontsize',fs)
legend('Nominal seeker','Perfect seeker','Interpreter','latex');
hold off;

subplot(1,2,2);
set(gca,'TickLabelInterpreter', 'latex')
hold on;
plot(t,x2,'LineWidth',i);
if i == 2
yline(s(2),'-.','LineWidth',1.2,'HandleVisibility','off')
end
grid on;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$x_2(t)$','Interpreter','latex', 'Fontsize',fs)
legend('Nominal seeker','Perfect seeker','Interpreter','latex');
hold off;



%% Figure 2 represents the forward velocity
% 
u1 = sqrt(w)*kt(x5,t);

if i == 2
set(gca,'TickLabelInterpreter', 'latex')
figure(3);
set(gca,'TickLabelInterpreter', 'latex')
subplot(1,2,1)
set(gca,'TickLabelInterpreter', 'latex')
plot(t,u1,'LineWidth',2);
yline(sqrt(w),'-.','LineWidth',2)
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$u_1(t) = \sqrt{\omega} \alpha(\eta(t))$','Interpreter','latex', 'Fontsize',fs)

set(gca,'TickLabelInterpreter', 'latex')
subplot(1,2,2)
set(gca,'TickLabelInterpreter', 'latex')
plot(t,x5,'LineWidth',2);
set(gca,'TickLabelInterpreter', 'latex')
grid on;
% axis equal;
xlabel( '$t$','Interpreter','latex','FontSize', fs)
ylabel('$\eta(t)$','Interpreter','latex', 'Fontsize',fs)
set(gca,'TickLabelInterpreter', 'latex')
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

dist(i) = norm(s - [x1(end),x2(end)]);

end

%% Distance between end point and source

dist
 

