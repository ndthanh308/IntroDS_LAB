def run_adaptive(self, x0: float, T: float, M: Callable, n: Callable, eps: float=1e-3) -> Tuple[float, float]:
        # (1)
        L = 0 
        Y, V, N = [], [], [] 
        Yl1, Yl2 = np.inf, np.inf
        beta = n(1)/n(0)
        convergence_err = np.inf
        # (2)
        while convergence_err > 0:
            N.append(10**3)
            # (3)
            YL, VL = self.__get_Yl(level=L, M=M, n=n, N=N[L], x0=x0, T=T)
            Y.append(YL)
            V.append(VL)
            _N = get_N(M, n, V, eps)
            # (4)
            for l in range(L+1):
                if _N[l] <= N[l]:
                    continue
                Yl, Vl = self.__get_Yl(level=l, M=M, n=n, N=(_N[l]-N[l]), x0=x0, T=T)
                Vl = (
                    (Vl + (_N[l]-N[l])/(_N[l]-N[l]-1)*Yl**2) * (_N[l]-N[l]-1) + 
                    (V[l] + N[l]/(N[l]-1)*Y[l]**2) * (N[l]-1)
                )/(_N[l]-1)
                Y[l] = (Yl*(_N[l]-N[l]) + Y[l]*N[l])/(_N[l])
                V[l] = Vl - (_N[l]/(_N[l]-1))*Y[l]**2
                N[l] = _N[l]
            Yl1 = Yl2
            Yl2 = YL
            L += 1
            # (5)
            convergence_err = max(
                abs(Yl2), abs(Yl1) / beta
            ) - (beta**(0.5)-1) * eps/np.sqrt(2)
        # (6)
        levels = np.arange(L)
        return np.sum(Y), np.sum(M(levels) * n(levels) * N)