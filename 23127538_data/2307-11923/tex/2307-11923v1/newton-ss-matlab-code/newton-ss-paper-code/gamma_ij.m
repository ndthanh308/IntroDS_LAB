function out2 = gamma_ij(ui,uj,k)
syms w p s t;

assume(w,'positive')

assume(t,'positive')


% k = 1/w^(1/2);
T = 2*pi/(w*k);

% T = T/2;

uj = subs(uj,t,s);
ui = subs(ui,t,p);

out1 = int(ui,p,0,s);

out2 = 1/T * int(uj*out1,s,0,T);

end
