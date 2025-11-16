%% Test

syms k F(x) a w w0 t k x ;

xdot = a * [cos(w*t - k*F(x));...
            sin(w*t - k*F(x))];
        
xdot = expand(xdot);


u1 = cos(k*F(x))*cos(3*w*t) + sin(k*F(x)) * sin(3*w*t);

xxdot = u1 * xdot;

simplify(expand(xxdot))