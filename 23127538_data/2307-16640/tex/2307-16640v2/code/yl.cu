__device__ FP sample_from_Yl(int ML, int nL,  int Ml, int nl, FP x0, FP T, curandState_t* state) {
    // (1)
    FP* dWL = (FP*)malloc(sizeof(FP) * ML);
    memset(dWL, 0, sizeof(FP) * ML);
    FP* dWl = (FP*)malloc(sizeof(FP) * Ml);
    memset(dWl, 0, sizeof(FP) * Ml);
    // (2)
    FP tL = 0;
    FP tl = 0;
    FP XL = x0;
    FP Xl = x0;
    // (3)
    int grid_density = LCM(nL, nl);
    FP H = T / grid_density;
    // (4)
    Jump* jumps_head = (Jump*)malloc(sizeof(Jump));
    generate_jumps<FP>(state, INTENSITY, T, jumps_head);
    Jump* jump_L = jumps_head;
    Jump* jump_l = jumps_head;
    // (5)
    FP t, dW;
    for (int i = 0; i < grid_density; i++) {
        t = (i+1)*H;
        // (6)
        for (int k = 0; k < ML; k++) {
            dW = (FP)(curand_normal(state) * sqrt(H));
            dWL[k] += dW;
            if (k < Ml) {
                dWl[k] += dW;
            }
        }
        // (7)
        if ((i+1)%(grid_density/nL) == 0) {
            jump_L = first_jump_after_time(jump_L, tL);
            XL = euler_single_step<FP>(tL, XL, t-tL, dWL, ML, jump_L, state);
            tL = t;
            memset(dWL, 0, sizeof(FP) * ML);
        }
        // (8)
        if ((i+1)%(grid_density/nl) == 0) {
            jump_l = first_jump_after_time(jump_l, tl);
            Xl = euler_single_step<FP>(tl, Xl, t-tl, dWl, Ml, jump_l, state);
            tl = t;
            memset(dWl, 0, sizeof(FP) * Ml);
        }
   }
   free_jumps_list(jumps_head);
   free(dWL);
   free(dWl);
   return func_f(XL) - func_f(Xl);
}