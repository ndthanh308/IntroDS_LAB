(* Created with the Wolfram Language : www.wolfram.com *)
-1 + 2*z - (-1 + z^(-1) + z/2)*Log[1 - z] - 2*z*Log[1 - z] + 
 (2 + z + 2*z*Log[1 - z])/4 - (2*z^2 + 2*z*(1 + z) + 2*z*(-3 + 4*z) + 
   (-3 + Sqrt[9 - 8*z] + 4*z)^2*ArcCoth[(3 - Sqrt[9 - 8*z] - 3*z)/z] + 
   (3 + Sqrt[9 - 8*z] - 4*z)^2*ArcCoth[(3 + Sqrt[9 - 8*z] - 3*z)/z] - 
   2*z^2*Log[1 - z] + 2*(1 + z^2)*Log[1 - z])/(8*z^2) + Log[(-1 + z)^2] + 
 (2*z - 2*z^2 + (3 - Sqrt[9 - 8*z] + 4*(-2 + Sqrt[9 - 8*z])*z + 8*z^2)*
    ArcCoth[(3 - Sqrt[9 - 8*z] - 3*z)/z] + 
   (3 + Sqrt[9 - 8*z] - 4*(2 + Sqrt[9 - 8*z])*z + 8*z^2)*
    ArcCoth[(3 + Sqrt[9 - 8*z] - 3*z)/z] - 
   4*ArcCoth[(2 - 2*z - Sqrt[8 - 8*z + z^2])/z] + 
   8*z*ArcCoth[(2 - 2*z - Sqrt[8 - 8*z + z^2])/z] - 
   10*z^2*ArcCoth[(2 - 2*z - Sqrt[8 - 8*z + z^2])/z] - 
   6*z*Sqrt[8 - 8*z + z^2]*ArcCoth[(2 - 2*z - Sqrt[8 - 8*z + z^2])/z] - 
   4*ArcCoth[(2 - 2*z + Sqrt[8 - 8*z + z^2])/z] + 
   8*z*ArcCoth[(2 - 2*z + Sqrt[8 - 8*z + z^2])/z] - 
   10*z^2*ArcCoth[(2 - 2*z + Sqrt[8 - 8*z + z^2])/z] + 
   6*z*Sqrt[8 - 8*z + z^2]*ArcCoth[(2 - 2*z + Sqrt[8 - 8*z + z^2])/z] + 
   3*Log[1 - z] - 2*z*Log[1 - z] - z^2*Log[1 - z] - 
   2*z*Log[(3 + Sqrt[9 - 8*z] - 2*z)/(2*z)] - 
   z^2*Log[(3 + Sqrt[9 - 8*z] - 2*z)/(2*z)] + 6*z*Log[z] + 3*z^2*Log[z] - 
   2*z*Log[(-3 + Sqrt[9 - 8*z] + 2*z)/(2*z)] - 
   z^2*Log[(-3 + Sqrt[9 - 8*z] + 2*z)/(2*z)] + 
   4*z*Log[(2 - z + Sqrt[8 - 8*z + z^2])/(2*z)] + 
   2*z^2*Log[(2 - z + Sqrt[8 - 8*z + z^2])/(2*z)] + 
   4*z*Log[(-2 + z + Sqrt[8 - 8*z + z^2])/(2*z)] + 
   2*z^2*Log[(-2 + z + Sqrt[8 - 8*z + z^2])/(2*z)])/(4*(-1 + z)*z) + 
 PolyLog[2, z] + PolyLog[2, z/(-1 + z)] + 
 z*(PolyLog[2, z] + PolyLog[2, z/(-1 + z)]) + 
 ((1 + z^2)*(Pi^2/12 + (PolyLog[2, z/(-1 + z)] + 
      PolyLog[2, (1 - (-3 - Sqrt[9 - 8*z] + 4*z)/(2*z))^(-1)] + 
      PolyLog[2, (1 - (-3 + Sqrt[9 - 8*z] + 4*z)/(2*z))^(-1)] - 
      2*PolyLog[2, (1 - (-2 + 3*z - Sqrt[8 - 8*z + z^2])/(2*z))^(-1)] - 
      2*PolyLog[2, (1 - (-2 + 3*z + Sqrt[8 - 8*z + z^2])/(2*z))^(-1)])/2))/
  (1 - z)
