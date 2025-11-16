#const wafers=25. 
route(1, 1, diffusion_fe_120, 20, 2, 4,       0). #(\label{prg:facts:prd:begin}#)
route(1, 2, lithotrack_fe_95,  6, 1, 1, su450_3).
route(1, 3,      implant_128,  8, 1, 1, su128_1).
route(1, 4, lithotrack_fe_95,  5, 1, 1, su450_3).
route(1, 5,      implant_128,  7, 1, 1, su128_2). #(\label{prg:facts:prd:end}#) 
setup(     implant_128, su128_1, 20, 4). #(\label{prg:facts:set:begin}#)
setup(     implant_128, su128_2, 18, 4). #(\label{prg:facts:set:mid}#)
setup(lithotrack_fe_95, su450_3, 22, 0). #(\label{prg:facts:set:end}#)
pm(     implant_128,      implant_128_mn, lots,   2,   3, 13). #(\label{prg:facts:pm:begin}#)
pm(lithotrack_fe_95, lithotrack_fe_95_wk, time,  12,  20, 15). 
pm(diffusion_fe_120, diffusion_fe_120_mn, time, 125, 150, 35). #(\label{prg:facts:pm:end}#)
tool(implant_128, 1). tool(lithotrack_fe_95, 1). tool(diffusion_fe_120, 1). #(\label{prg:facts:tool:begin}#) #(\label{prg:facts:tool:end}#)
lot(1, 1). lot(2, 1). #(\label{prg:facts:wip:begin}#) #(\label{prg:facts:wip:end}#)
