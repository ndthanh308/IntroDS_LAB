
function out3 = gamma_ijm(ui,uj,um,k)
syms w p s t tau;

assume(w,'positive')

assume(t,'positive')

% k = 1/w^(1/2);
T = 2*pi/(w*k);

% T = T/2;
ujs = subs(uj,t,s);
uip = subs(ui,t,p);

uis = subs(ui,t,s);
ujp = subs(uj,t,p);

umtau = subs(um,t,tau);


o1 = int(ujs * uip - uis * ujp,p,0,s);

o2 = int(o1,s,0,tau);

out3 = 1/(3*T) * int(umtau*o2, tau, 0, T);
end