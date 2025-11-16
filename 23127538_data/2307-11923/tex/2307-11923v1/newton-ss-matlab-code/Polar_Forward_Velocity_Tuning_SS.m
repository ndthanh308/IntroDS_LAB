%% Unicycle Source Seeking via Forward Velocity Tuning - Calculations
%% Dynamics
clc;
clear;

syms w t a c Fm r y h;

w0 = w/4;

J =  Fm - 0.5 * h*r^2; 



v = c * J *sin(w*t) + a * w * cos(w * t);



syms tau gamma;

rdot = gamma * v * cos( w0*t - y);

ydot = gamma * v * sin(w0*t - y )/r;

rdot = expand(rdot);

ydot = expand(ydot);


% r_bar = r - a*sin(w*t)*cos(w0*t);
% y_bar = y - a * sin(w*t) * sin(w0*t);


J = subs(J,r,r + a*sin(w*t)*cos(w0*t));

rdot = subs(rdot,r,r + a*sin(w*t)*cos(w0*t));

ydot = subs(ydot,r,r + a*sin(w*t)*cos(w0*t));

% ydot = subs(ydot,y,y + a*sin(w*t)*sin(w0*t));

r_bar_dot = rdot - diff(a*sin(w*t)*cos(w0*t),t);

y_bar_dot = ydot;


% xdot = expand(xdot)
% xdotavg = 1/(2*pi) * int(xdot,tau,0,2*pi)

 syms tau;
 
 r_bar_dot = 1/w * subs(r_bar_dot,w*t,tau);
 
 y_bar_dot = 1/w * subs(y_bar_dot,w*t,tau);
 
 r_bar_dot = subs(r_bar_dot,tau,4*tau);
 
 y_bar_dot =  subs(y_bar_dot,tau,4*tau);

%%
r_bar_dot = expand(r_bar_dot);
R = children(r_bar_dot);

assume(R,'real')
rdotavg = 0;
for i = 1:length(R)
%   disp(['i is: ', num2str(i)])
 rdotavg = rdotavg +  vpa(1/(2*pi) * int(R(i),tau,0,2*pi),2);
end

rdotavg


%%
% y_bar_dot = expand(y_bar_dot);
% Y = children(y_bar_dot);
% 
% assume(Y,'real')
% ydotavg = 0;
% for i = 1:length(Y)
% %   disp(['i is: ', num2str(i)])
%  ydotavg =  ydotavg + vpa(1/(2*pi) * int(Y(i),tau,0,2*pi),2);
% end
% 
% ydotavg



%%
syms wr;
J = subs(J,w*t,tau);
J = subs(J,tau,4*tau);
%%
Ny = wr/w*(gamma -  gamma^2 * 16/a^2 * cos(2*tau)* J);


Ny = expand(Ny);


Z = children(Ny);


% ydotavg = zeros(length(Y),1);
gammadotavg = 0;
for i = 1:length(Z)
%   disp(['i is: ', num2str(i)])
 gammadotavg = gammadotavg + vpa(1/(2*pi) * int(Z(i),tau,0,2*pi),2);
end

gammadotavg

