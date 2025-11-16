(* ::Package:: *)

(* Created with the Wolfram Language : www.wolfram.com *)
HfCFTF[Z_,V_]:=Module[{SQRT,z=Z,v=V},

SQRT=Sqrt[(1+(-1+z) v^2) (v^2-v^4+z (-2+v^2)^2)];

-(26 - 16/z + (2*v^2*(1 + v^2*(-1 + z) - 2*z)*(-3 + v^4*(-1 + z)^2 + 
       2*(-1 + z)*z - 2*v^2*(-1 + z)*z))/(SQRT*(v^2*(-1 + z) - z)) + 
    v*(-22 + 16/z + 18*z + 7*z^2 - 4*z^3) + z*(-14 + z*(-9 + 4*z)))/
  (4*(1 - z)) + ((8 + 2*(-8 + z)*z - v*(10 + (-14 + z)*z))*Log[2*(1 - v)])/
  (2*(-1 + z)) + ((2*v + 2*z^2 - v*z*(2 + z))*Log[z])/(2*(-1 + z)) + 
 ((v^3 + 2*v*z - 2*v^3*z)*Log[(SQRT + v - v^3 + 2*v*z - v^3*z)/
     (SQRT - v + v^3 - 2*v*z + v^3*z)])/(2 - 2*z) + 
 (v*(-1 + z)*(-2*z + v^2*(-1 + 2*z))*
   Log[(SQRT*v^2 - v^3 + v^5 + SQRT*z - 3*v*z - SQRT*v^2*z + 4*v^3*z - 
      2*v^5*z + 2*v*z^2 - 3*v^3*z^2 + v^5*z^2)/(SQRT*v^2 + v^3 - v^5 + 
      SQRT*z + 3*v*z - SQRT*v^2*z - 4*v^3*z + 2*v^5*z - 2*v*z^2 + 3*v^3*z^2 - 
      v^5*z^2)])/(2*(-(v^2*(-1 + z)) + z)^2)
 ]
