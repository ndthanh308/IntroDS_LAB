%% Simulation

clc;
clear;
% clf;
hold on;
scalesize = 0.75;
latex_fs = 12;
fs = latex_fs/scalesize;
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontSize', fs)
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontSize', fs)
set(0,'DefaultAxesFontName','CMU Serif')

%% Parameters
p = 0.61;  % p \in (0.5, 1) 

w = 15;
w0 = 1;
a = 2;
c = 1;
h = 1;
% b = 3;

w_d = 0.3;

H = 1/100;
x_star = 1;
y_star = -1;
F_star = 5;

F = @(x,y) F_star - 1/2 * H * ((x-x_star).^2 + (y-y_star).^2); 

x10 = 4;
x20 = -4;

% J0 = [0, w0;-w0 0];
% for j = 1:2
%     if j == 2
%     w0 = 3;
%     J0 = [0, w0;-w0 0];
%     end
%% System
for i = 1:2
    
if i == 1
% Gradient Seeker
f = @(t,x) [ a*w^(p)*cos(t*w)*cos(t*w0) + c*w^(1-p)*sin(t*w)*cos(t*w0)*(F(x(1),x(2)) - x(3));...
             a*w^(p)*cos(t*w)*sin(t*w0) + c*w^(1-p)*sin(t*w)*sin(t*w0)*(F(x(1),x(2)) - x(3));...
                                                                     h*(F(x(1),x(2)) - x(3)) ];
                                        
x0 = [x10;x20;0];    

end
if i == 2    

% System that estimates the inverse of the Hessian directly
f = @(t,x) [ a*w^(p)*cos(t*w)*cos(t*w0) + c*x(4)*w^(1-p)*sin(t*w)*cos(t*w0)*(F(x(1),x(2)) - x(3));...
             a*w^(p)*cos(t*w)*sin(t*w0) + c*x(4)*w^(1-p)*sin(t*w)*sin(t*w0)*(F(x(1),x(2)) - x(3));...
                                                                          h*(F(x(1),x(2)) - x(3));...
                          w_d*  x(4) * (1 - x(4)*8*w^(2-2*p)/a^2*(F(x(1),x(2)) - x(3))*cos(2*w*t))];
                                        
% x0 = [4;-4;0;2;-0.4]; 
% g = @(t,x)  [ J0*[x(1);x(2)] + ( a*w^(p)*cos(t*w) + c*x(4)*w^(1-p)*sin(t*w)*(F(x(1),x(2)) - x(3) ) )*[0;1];...
%                                                                                  h*(F(x(1),x(2)) - x(3));...
%                                 w_d*  x(4) * (1 - x(4)*8*w^(2-2*p)/a^2*(F(x(1),x(2)) - x(3))*cos(2*w*t))];
x0 = [x10;x20;0;1];   
end                                        
Tsim = 50;
tspan = [0,Tsim]; 

 
sol = ode15s(f, tspan, x0);
% if i == 2
% sol_g = ode15s(g,tspan,x0);
% end
t = sol.x;
x1 = sol.y(1,:);
x2 = sol.y(2,:);
x4 = zeros(size(x1));
% x5 = x4;
if i == 2
x3 = sol.y(3,:);
x4 = sol.y(4,:);
% x5 = sol.y(5,:);
end
% x4 = sol.y(4,:);
X_data{i} = [t.',x1.',x2.',F(x1,x2).',x4.'];


end


% movegui(f1,'southeast');
% movegui(f2,'west');
% movegui(f3,'east');

%% Export plots
X_data_grad = X_data{1};

t_grad = X_data_grad(:,1);
x1_grad = X_data_grad(:,2);
x2_grad = X_data_grad(:,3);
F_grad = X_data_grad(:,4);

X_data_new = X_data{2};

t_new = X_data_new(:,1);
x1_new = X_data_new(:,2);
x2_new = X_data_new(:,3);
F_new = X_data_new(:,4);
d = X_data_new(:,5);

% t_z = sol_g.x;
% z1 = sol_g.y(1,:);
% z2 = sol_g.y(2,:);
%%
opt.fname = 'newton_ss_data';
% data2txt(opt, t_grad, x1_grad, x2_grad, F_grad,...
%               t_new, x1_new, x2_new, F_new, d);
                                                       

%% Plots New
purp = [0.4940 0.1840 0.5560];
green = [0.4660 0.6740 0.1880];

%% Plot Trajectories
% hold on;
% figure(1);
% p1 = plot(x1_grad,x2_grad,'LineWidth',1.5,'Color',green);
% p2 = plot(x1_new,x2_new,'LineWidth',2,'Color',purp);
% plot(x_star,y_star,'x','LineWidth',2,'Color',[0,0,0],'MarkerSize',10);
% xlabel('$x_1$','Interpreter','latex')
% ylabel('$x_2$','Interpreter','latex')
% grid on;
% axis equal;
% legend('Gradient seeker (6)','Newton seeker (23)','Source $x^* = [1,-1]^T$', 'Interpreter','latex','location','southeast');
% % ylim([4.82,5.02])
% box on;

%% F-Plot
x0 = 1;
y0 = 1;
width = 12;
height = 9;
figure('Units','inches',...
    'PaperSize',[8.5,11],...
'Position',[x0 y0 width height],....
'PaperPositionMode','auto');


set(gca,'TickLabelInterpreter', 'latex','FontSize',100/3);
hold on;
plot(t,F_star + 0.*t,'-.','LineWidth',2,'Color',[0,0,0],'HandleVisibility','off');
p1 = plot(t_grad,F_grad,'LineWidth',1.5,'Color',green);
p2 = plot(t_new,F_new,'LineWidth',2,'Color',purp);

xlabel('$t$','Interpreter','latex')
ylabel('$F(x)$','Interpreter','latex')
grid on;
legend('Gradient seeker','Newton seeker', 'Interpreter','latex','location','southeast');
ylim([4.82,5.02])
box on;
 print -depsc2 field_newton.eps
%% States x1
x0 = 1;
y0 = 1;
width = 12;
height = 9;
figure('Units','inches',...
    'PaperSize',[8.5,11],...
'Position',[x0 y0 width height],....
'PaperPositionMode','auto');


set(gca,'TickLabelInterpreter', 'latex','FontSize',100/3);
hold on;
plot(t,1 + 0.*t,'-.','LineWidth',2,'Color',[0,0,0],'HandleVisibility','off');
p1 = plot(t_grad,x1_grad,'LineWidth',1.5,'Color',green);
p2 = plot(t_new,x1_new,'LineWidth',2,'Color',purp);

xlabel('$t$','Interpreter','latex')
ylabel('$x_1(t)$','Interpreter','latex')
grid on;
legend('Gradient seeker','Newton seeker', 'Newton Seeker in $z$','Interpreter','latex','location','northeast');
% ylim([4.82,5.02])
box on;
print -depsc2 x1.eps
%% States x2
x0 = 1;
y0 = 1;
width = 12;
height = 9;
figure('Units','inches',...
    'PaperSize',[8.5,11],...
'Position',[x0 y0 width height],....
'PaperPositionMode','auto');


set(gca,'TickLabelInterpreter', 'latex','FontSize',100/3);
hold on;
plot(t,-1 + 0.*t,'-.','LineWidth',2,'Color',[0,0,0],'HandleVisibility','off');
p1 = plot(t_grad,x2_grad,'LineWidth',1.5,'Color',green);
p2 = plot(t_new,x2_new,'LineWidth',2,'Color',purp);
xlabel('$t$','Interpreter','latex')
ylabel('$x_2(t)$','Interpreter','latex')
grid on;
legend('Gradient seeker','Newton seeker', 'Newton Seeker in $z$','Interpreter','latex','location','southeast');
ylim([-5.5,0])
box on;
print -depsc2 x2.eps

%% Hessian
% 
% x0 = 1;
% y0 = 1;
% width = 12;
% height = 9;
% figure('Units','inches',...
%     'PaperSize',[8.5,11],...
% 'Position',[x0 y0 width height],....
% 'PaperPositionMode','auto');
% 
% 
% set(gca,'TickLabelInterpreter', 'latex','FontSize',100/3);
% hold on;
% plot(t,1/H + 0.*t,'-.','LineWidth',2,'Color',[0,0,0],'HandleVisibility','off');
% p1 = plot(t_new,d,'LineWidth',1.5,'Color',green);
% 
% 
% xlabel('$t$','Interpreter','latex')
% ylabel('$d(t)$','Interpreter','latex')
% grid on;
% % legend('Gradient seeker (6)','Newton seeker (23)', 'Interpreter','latex','location','southeast');
% % ylim([-5.5,0])
% box on;


%% For Showing to Miro

% if j == 1
% subplot(1,2,1);
% hold on;
% plot(t,-1 + 0.*t,'-.','LineWidth',2,'Color',[0,0,0],'HandleVisibility','off');
% % p1 = plot(t_grad,x1_grad,'LineWidth',1.5,'Color',green);
% p3 = plot(t_z,z2,'LineWidth',2,'Color',green);
% p2 = plot(t_new,x2_new,'LineWidth',2,'Color',purp);
% 
% xlabel('$t$','Interpreter','latex')
% ylabel('$x_1(t)$','Interpreter','latex')
% grid on;
% legend('Newton seeker (23)', 'Newton Seeker in $z$ (27)','Interpreter','latex','location','northeast');
% % ylim([4.82,5.02])
% box on;
% title('$\omega_0 = 0.3$')
% end
% if j ==2
% subplot(1,2,2)
% hold on;
% plot(t,-1 + 0.*t,'-.','LineWidth',2,'Color',[0,0,0],'HandleVisibility','off');
% % p1 = plot(t_grad,x2_grad,'LineWidth',1.5,'Color',green);
% p2 = plot(t_new,x2_new,'LineWidth',2,'Color',purp);
% p3 = plot(t_z,z2,'LineWidth',2,'Color',green);
% xlabel('$t$','Interpreter','latex')
% ylabel('$x_2(t)$','Interpreter','latex')
% grid on;
% legend('Newton seeker (23)', 'Newton Seeker in $z$ (27)','Interpreter','latex','location','southeast');
% % ylim([-5.5,0])
% box on;
% title('$\omega_0 = 1.5$')
% end
% 
% end

% sgtitle('Trajectories for $\omega_0 = 2$','Interpreter','latex') 