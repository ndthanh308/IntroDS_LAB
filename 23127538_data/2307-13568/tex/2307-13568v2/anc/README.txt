Description of supplemental files
_____________________________________________________________________________________

gauss_parameters.csv:

Coefficients used for approximating the central NN potentials up to different chiral
orders by sums of Gaussians according to Eqs. (7) and (8). These approximations are
used when determining Hartree energies from long-range NN forces. At LO there is no
Hartree contribution from central long-range NN forces. Values marked with * are
determined from the other coefficients following Eq. (10). mu_i are given in fm; W_i
and H_i are in MeV.

Format: Parameter,i=1,i=2,i=3,i=4,i=5

_____________________________________________________________________________________

nn_dme_interpolation_parameters.csv:

Coefficients used for approximating the NN g coefficient functions at different
chiral orders according to Eq. (17). These approximations are used when determining
Fock energies from long-range NN forces. g(0) and a_i are given in MeV*fm^3 for the
g_t^RhoRho coefficients and in MeV*fm^5 for the other coefficients; b_i are in
fm^(3*c_i); c_i are unitless.

Format: Order,Coefficient,g(0),a_1,b_1,c_1,a_2,b_2,c_2,a_3,b_3,c_3

_____________________________________________________________________________________

3n_dme_interpolation_parameters.csv:

Coefficients used for approximating the 3N g coefficient functions at different
chiral orders according to Eq. (18). These approximations are used when determining
Fock energies from long-range 3N forces. g(0) and a_i are given in MeV*fm^6 for the
g^Rho0Rho0Rho0 and g^Rho0Rho1Rho1 coefficients and in MeV*fm^8 for the other
coefficients; b_i are in fm^(3*c_i); c_i are unitless.

Format: Order,Coefficient,g(0),a_1,b_1,c_1,a_2,b_2,c_2,a_3,b_3,c_3

_____________________________________________________________________________________

inm_properties_edf_parameters.csv:

Parameters of the different GUDE variants obtained in this work. The first block
contains infinite nuclear matter properties, the second block contains volume
parameters, the third block contains surface and pairing parameters. rho_c is given
in fm^-3; E_sat, K, a_sym, and L_sym are in MeV; Ms*^-1 and gamma are unitless;
C_t0^RhoRho and V_0^q are in MeV*fm^3; C_tD^RhoRho are in MeV*fm^(3+3*gamma);
C_t^RhoTau, C_t^RhoDeltaRho, C_t^RhoNablaJ, and C_t^JJ are in MeV*fm^5.

Values are given in every row for the different GUDE variants in the order:
no_chiral,LO,NLO,N2LO,N2LO+3N,NLOD,NLOD+3N,N2LOD,N2LOD+3N,min_chiral
