% NLPsoft.mp -- do not edit, generated automatically by output.tex
input feynmp
require_RCS_revision "1.30";
 path quadrant, q[], otimes; quadrant = (0, 0) -- (0.5, 0) & quartercircle & (0, 0.5) -- (0, 0); for i=1 upto 4: q[i] = quadrant rotated (45 + 90*i); endfor otimes = q[1] & q[2] & q[3] & q[4] -- cycle; 
feynmfwizard := true;
beginchar(1, 120*1.0pt#, 100*1.0pt#, 0);
"feynmf: 1";
LaTeX_unitlength:=1.0pt;
subgraph (0, 0, w, h);
vinit;
pickup pencircle scaled thin;
straight_galleries;
vleft (__a1, __a2, __a3, __a4, __a5, __a6, __a7, __a8, __a9);
vright (__m1, __m2, __m3, __m4, __m5, __m6, __m7, __m8, __m9);
vconnect ("phantom", __a1, __b1, __c1, __d1, __e1, __f1, __g1, __h1, __i1, __j1, __k1, __l1, __m1);
vconnect ("phantom", __a2, __b2, __c2, __d2, __e2, __f2, __g2, __h2, __i2, __j2, __k2, __l2, __m2);
vconnect ("phantom", __a3, __b3, __c3, __d3, __e3, __f3, __g3, __h3, __i3, __j3, __k3, __l3, __m3);
vconnect ("phantom", __a4, __b4, __c4, __d4, __e4, __f4, __g4, __h4, __i4, __j4, __k4, __l4, __m4);
vconnect ("phantom", __a5, __b5, __c5, __d5, __e5, __f5, __g5, __h5, __i5, __j5, __k5, __l5, __m5);
vconnect ("phantom", __a6, __b6, __c6, __d6, __e6, __f6, __g6, __h6, __i6, __j6, __k6, __l6, __m6);
vconnect ("phantom", __a7, __b7, __c7, __d7, __e7, __f7, __g7, __h7, __i7, __j7, __k7, __l7, __m7);
vconnect ("phantom", __a8, __b8, __c8, __d8, __e8, __f8, __g8, __h8, __i8, __j8, __k8, __l8, __m8);
vconnect ("phantom", __a9, __b9, __c9, __d9, __e9, __f9, __g9, __h9, __i9, __j9, __k9, __l9, __m9);
vconnect ("dashes,tension=0", __g1, __g9);
vconnect ("gluon,tension=0", __c7, __m5);
vconnect ("double,tension=0", __a5, __e9);
vconnect ("double,tension=0", __a5, __e1);
vconnect ("double,tension=0", __m5, __i9);
vconnect ("double,tension=0", __m5, __i1);
vvertex ("d.sh=otimes,d.f=empty,d.si=.1w", __m5);
vfreeze;
vdraw;
endsubgraph;
endchar;
% the end.
end.
endinput;
