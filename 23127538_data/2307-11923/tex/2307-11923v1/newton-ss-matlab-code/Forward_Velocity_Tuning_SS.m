%% Unicycle Source Seeking via Forward Velocity Tuning - Calculations

clc;
clear;

syms w0 w t a c q x y Fm xm ym;

J =  Fm - q*(x-xm)^2 - q*(y-ym)^2 ; 

v = c * J *sin(w*t) + a * w * cos(w * t);

%% Dynamics

syms tau gamma e;

k = 4;
Jm = Fm - q* (x + a * sin(k *tau) * cos(tau) )^2 - q * (y   + a * sin(k*tau) * sin(tau))^2;
delta = - q * (x + a * sin(k *tau) * cos(tau) )^2 - q * (y   + a * sin(k*tau) * sin(tau))^2  - e;

xdot = 1/w  * gamma * (c * delta * sin(k*tau) * cos(tau) + a * w0 * sin(k*tau) * sin(tau) );

ydot = 1/w  * gamma * (c * delta  * sin(k*tau) * sin(tau) - a * w0 * sin(k*tau) * cos(tau) );


%% 

% assume(in(k,'integer') & k>3)
% 
% k = 4;
xdot = expand(xdot);
% xdotavg = 1/(2*pi) * int(xdot,tau,0,2*pi)
ydot = expand(ydot);
 

%%

X = children(xdot);

 

 
assume(X,'real')
xdotavg = 0;
for i = 1:length(X)
%   disp(['i is: ', num2str(i)])
 xdotavg =  xdotavg + vpa(1/(2*pi) * int(X(i),tau,0,2*pi),2);
end

xdotavg

%%


Y = children(ydot);

 

 
assume(Y,'real')
ydotavg = 0;

for i = 1:length(Y)
%   disp(['i is: ', num2str(i)])
 ydotavg =  ydotavg + vpa(1/(2*pi) * int(Y(i),tau,0,2*pi),2);
end

ydotavg
%%
syms wr u;
% clc;
Ny = wr*( gamma - gamma^2 * 4/a^2 * cos(2*k*tau) * Jm);


Ny = expand(Ny);

assume(Ny,'real');
NY = children(Ny);


% ydotavg = zeros(length(Y),1);
% gamma_avg = 0;
clear gamma_avg;
for i = 1:length(NY)
%   disp(['i is: ', num2str(i)])
 gamma_avg(i,:)  =   vpa(1/(2*pi) * int(NY(i),tau,0,2*pi),2);
end

vpa(sum(gamma_avg),2)


