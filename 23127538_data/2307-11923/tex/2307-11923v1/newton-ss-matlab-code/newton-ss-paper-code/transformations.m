syms w0 a H t h;

J0 = [0,w0;-w0 0];
L = [0 0;0 -a*H/2];

Y = [sin(w0*t) cos(w0*t);-cos(w0*t) sin(w0*t)];

P = J0.' + Y*(J0 + L)*Y.'


JJ = [0 w0 0;-w0 -a*H/2 0;0 0 -h]


%% 
w0 = 0.2;
a = 2;

lam = -a/2 + 1/2 * sqrt((a/2)^2 - (2*w0)^2)

