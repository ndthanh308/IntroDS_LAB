clc;
clear all;
close all;

%% KF-based instantaneous phase and frequency offset tracking

% process model
dphi_1 = 0;
df_1 = 1000;
dw_1 = 2*pi*df_1;
sim_time = 1000;
T = 50e-3;
dphi = zeros(sim_time,1);
dw = zeros(sim_time,1);
n = zeros(2,sim_time);
wc = 2*pi*900e6;
q1 = 8.47e-22; % for USRP N200's
q2 = 5.51e-18;
mu = [0 0];
Q = wc^2*q1*[T 0; 0 0] + wc^2*q2*[T^3/3 T^2/2; T^2/2 T];
n = mvnrnd(mu, Q, sim_time);
dphi(1) = dphi_1;
dw(1) = dw_1;

for k=2:sim_time
    dphi(k) = dphi(k-1) + T*dw(k) + n(k,1);
    dw(k) = dw(k-1) + n(k,2);
end

t = T*(0:1:sim_time-1);
% subplot(2,1,1);
% plot(t,dphi-dw.*t');
% xlabel('time (sec)');
% ylabel('unwrapped instantaneous phase offset (radians)');
% subplot(2,1,2);
% plot(t, dw./(2*pi));
% xlabel('time (sec)');
% ylabel('instantaneous freq offset (Hz)');

%% measurement model
% process model (at Alice)
dw_A = dw + 2*randn(sim_time,1);

% process model (at Bob)
dw_B = dw + 2*randn(sim_time,1);

figure;
plot(t, dw_A./(2*pi));
xlabel('time (sec)');
ylabel('instantaneous freq offset (Hz)');
hold on;
plot(t,dw_B./(2*pi),'r');
legend('as seen by Alice','as seen by Bob');
