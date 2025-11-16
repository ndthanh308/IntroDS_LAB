// Returns next point of trajectory
template <typename type>
__device__ type euler_single_step(type ti, // previous grid point
                                  type Xi, // previous trajectory value                                 
                                  type hl, // stepsize
                                  type* dWl, // multidim Wiener increment
                                  int Ml,  // dimension of Wiener increment
                                  Jump* jumps, // list of jumps starting from ti
                                  curandState_t* state) {
    type X = Xi;
    
    // Adding 'a' coefficient
    X += func_a<type>(ti + ((type)curand_uniform_double(state)) * hl, Xi) * hl;
    
    // Adding 'b' coefficient
    for (int k = 0; k < Ml; k++) {
        X += func_b<type>(k + 1, ti, Xi) * dWl[k];
    }
    
    // Adding 'c' coefficient
    if (jumps != NULL) {
        while (jumps->time < ti+hl) {
            if (jumps->time > ti) {
                X += func_c<type>(ti, Xi, jumps->height);
            }
            jumps = jumps->next_jump;
            if (jumps == NULL) break;
        }
    }
    
    // Next sample from single Euler step
    return X;
}