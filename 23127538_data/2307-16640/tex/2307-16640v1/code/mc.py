def run_single_level(
        self,
        x0: float, 
        T: float, 
        M0: int, 
        n0: int,
        N0: int
    ) -> Tuple[float, float]:
        """
        Runs standrad Monte Carlo simulation with 
        given initial conditions and parameters.
        Returns algorithm result and informational cost.
        """
        Y0, V0 = self.__get_Yl(
            level=0, 
            M=lambda l: M0, 
            n=lambda l: n0, 
            N=N0, 
            x0=x0, 
            T=T
        )
        return np.sum(Y0), M0*n0*N0