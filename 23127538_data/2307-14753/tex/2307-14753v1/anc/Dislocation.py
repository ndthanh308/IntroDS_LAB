# Copyright (c) 2022-2023, Nisarg Chadha, Ali G. Moghaddam, 
#                          Jeroen van den Brink, Ion Cosma Fulga. 
#                          All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     2) Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
----------------------------------------------------------------
Real-space topological localizer index to fully characterize the
                    dislocation skin effect
----------------------------------------------------------------

In this module we reproduce some of the results presented in the paper:

N. Chadha, A. G. Moghaddam, J. v. d. Brink, I. C. Fulga
"Real-space topological localizer index to fully characterize the 
                                          dislocation skin effect"
arXiv:XXXX.XXXXX.

For more information on kwant:

https://kwant-project.org/

C.W. Groth, M. Wimmer, A.R. Akhmerov and X. Waintal
"kwant: a software package for quantum transport"
arXiv:1309.2916.

For examples of usage, see the main() function, which reproduces
some of our numerical results. This script can be imported in a
python interface or simply run as:

python3 Dislocation.py

"""

import kwant
from kwant.digest import uniform
import numpy as np
import pylab as py

py.ion()

p = dict(L=70, W=70, d=50, tx=1, ty=0.4, mu=0, ex=0.4, ey=0, PBCx=1, PBCy=1,
         onsdis=0, salt='1237')

lat = kwant.lattice.square()

def hop_x(site0, site1, tx, ex):
    '''Hopping in the x-direction'''
    return tx*(1-1j*ex)


def hop_x_PBC(site0, site1, tx, ex, PBCx):
    '''Hopping in the x-direction under PBC'''
    return PBCx * tx * (1 - 1j * ex)


def hop_y(site0, site1, ty, ey):
    '''Hopping in the y-direction'''
    return ty * (1 - 1j * ey)


def hop_y_PBC(site0, site1, ty, ey, PBCy):
    '''Hopping in the y-direction under PBC'''
    return PBCy * ty * (1 - 1j * ey)


def onsite(site, mu, onsdis, salt):
    '''On-site potential'''
    onsdis = (uniform(repr(site), salt) - 0.5) * onsdis
    return mu + onsdis


def build_system(L=60, W=30, d=10, B=1):
    '''Constructs the model for the weak Hatano-Nelson Model with a pair of
       dislocations.

    Parameters
    ----------
    L : integer
        Number of sites along x
    W : integer
        Number of sites along y
    d : integer
        Distance between the two point-defects
    B : integer
        Burger's vector of the dislocation

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        Finalized system
    '''

    syst = kwant.Builder()
    for i in range(L):
        for j in range(W):
            syst[lat(i, j)] = onsite

    syst[kwant.HoppingKind((1, 0), lat, lat)] = hop_x
    syst[kwant.HoppingKind((-(L-1), 0), lat, lat)] = hop_x_PBC

    syst[kwant.HoppingKind((0, 1), lat, lat)] = hop_y
    syst[kwant.HoppingKind((0, -(W-1)), lat, lat)] = hop_y_PBC

    for j in range(B):
        for i in range(d):
            del syst[lat(L//2 - d//2 + i, W//2-B//2+j)]

    for i in range(d):
        syst[lat(L//2-d//2+i, W//2-B//2-1),
             lat(L//2-d//2+i, W//2-B//2+B)] = hop_y

    return syst.finalized()


def build_ham(syst):
    '''Builds the Hamiltonian for the system

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
           System whose Hamiltonian needs to be computed

    Returns
    -------
    ham : 2d array-like
          Hamiltonian for the system
    '''

    Hr = syst.hamiltonian_submatrix(params=p).real
    Hi = syst.hamiltonian_submatrix(params=p).imag
    ham = Hr - Hi
    return ham


def plot_spd(syst):
    '''Computes and plots the density of states for the doubled Hamiltonian

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
    '''
    ham = build_ham(syst)
    eigval, eigvec = np.linalg.eig(ham)
    pos = syst.sites
    xpos = np.zeros([len(pos)])
    ypos = np.zeros([len(pos)])
    for i in range(len(pos)):
        xpos[i] = pos[i].tag[0]
        ypos[i] = pos[i].tag[1]

    ldos = np.zeros(len(eigval))
    for i in range(len(eigval)):
        vec = eigvec[:, i]
        vec = vec/np.linalg.norm(vec)
        ldos += abs(vec)**2

    lldos = np.log(ldos)
    fig = py.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'x', fontsize=18)
    ax.set_ylabel(r'y', fontsize=18)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\ln{\rho}$', fontsize=18)
    ax.tick_params('z', labelsize=15)
    ax.tick_params('x', labelsize=15)
    ax.tick_params('y', labelsize=15)
    dos = ax.plot_trisurf(xpos, ypos, lldos, cmap="viridis", rasterized=True)
    py.colorbar(dos)


def build_doubled_ham(syst):
    '''Builds the doubled Hamiltonian

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem

    Returns
    -------
    herm_ham : 2d array-like
               Hamiltonian for the doubled system
    '''
    ham = build_ham(syst)
    herm_ham = np.block([[ ham * 0,     ham ],
                         [ ham.T  , ham * 0 ]])

    return herm_ham


def plot_doubled_spectrum(syst):
    '''Plots the eigenspectrum for the doubled Hamiltonian'''

    herm_ham = build_doubled_ham(syst)
    eigval = np.linalg.eigvalsh(herm_ham)

    py.figure()
    py.title("Eigenvalues of the doubled hamiltonian")
    py.ylabel("Energy Eigenvalue")
    py.plot(eigval, 'o')


def plot_ldos(syst, edge_threshold):
    '''Plots the density of zero modes of the hermitian system
    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
           Original non-Hermitian system
    edge_threshold : float
                     Tolerance for a mode to be classified as a zero mode
    '''
    herm_ham = build_doubled_ham(syst)
    eigval, eigvec = np.linalg.eigh(herm_ham)
    edge_count = 0
    N = len(herm_ham)//2
    ldos = np.zeros(N)

    for i in range(len(eigval)):
        if abs(eigval[i]) < edge_threshold:
            edge_count += 1
            ldos += np.abs(eigvec[:, i][:N])**2 + np.abs(eigvec[:, i][N:])**2

    print("Number of zero modes=%s" % edge_count)
    fig,ax=py.subplots()
    kwant.plotter.map(syst, ldos, vmax=max(ldos),ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def get_gap_index(syst, B, Lin, Lout):
    '''Calculates the energy gap and topological index for a system with 
       disorder

    Parameters
    ----------
    syst : kwant.builder.FinalSystem
           Original non-Hermitian system
    B : integer
        Burger's vector
    Lin : Position taken to be between the point defects
    Lout : Position taken to be outside the point defects

    Returns
    -------
    gap : float
          Bulk energy gap for the hermitian hamiltonian
    topind : integer
             Value of the topological index with the value outside the 
                                                dislocation subtracted
    '''
    ham = build_ham(syst)
    herm_ham = build_doubled_ham(syst)
    eigval = np.linalg.eigvalsh(herm_ham)
    
    gap = 2 * abs(eigval[len(herm_ham)//2 + B])
    topind = top_ind(syst, ham, Lin) - top_ind(syst, ham, Lout)
    return gap, topind


def top_ind(syst, origin):
    '''Calculates the localiser index of the system with respect to a 
       specified origin

    Parameters
    ----------
    syst : kwant.builder.FinalSystem
           Original non-Hermitian system
    origin : float
             Position of the origin(non-integer) for computing the 
             position operator
    '''

    ham = build_ham(syst)

    pos = syst.sites
    xpos = np.zeros([len(pos)])
    for i in range(len(pos)):
        xpos[i] = pos[i].tag[0] - origin

    X = np.diag(xpos)
    Xinv = np.diag(1/xpos)

    M = X + ham.T @ Xinv @ ham
    
    l = np.linalg.eigvalsh(M)
    sig = np.sum(np.sign(X)) - np.sum(np.sign(l))
    sig = sig/2
    return sig

def plot_topind(syst, originrange):
    '''Plots the variation of the topological index as a function of 
       the origin

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
           Original non-Hermitian system
    originrange : 1d-array like
                  Range of origin values for computing the index
    '''

    indlist = []

    for origin in originrange:
        indlist.append(top_ind(syst, origin))

    py.figure()
    py.xlabel(r'$x_0$')
    py.ylabel("Topological index")
    py.plot(originrange, indlist, marker="o")


def plot_gap_index_disorder():
    '''Plots the bulk energy gap and the topological index with appropriate 
       error bars for a disordered system averaged over N=100 samples

       Data is stored in the auxiliary file disorder_data.txt'''
       
    data = np.loadtxt("disorder_data.txt", delimiter=",")
    disorder = data[:, 0]
    gap = data[:, 1]
    gap_low = data[:, 2]
    gap_up = data[:, 3]
    ind = data[:, 4]
    ind_low = data[:, 5]
    ind_up = data[:, 6]
    py.figure()
    py.xlabel('W')
    py.errorbar(disorder, gap, yerr=(gap_low,gap_up), label='Bulk Gap')
    py.errorbar(disorder, ind, yerr=(ind_low,ind_up), label='Topological Index')
    py.legend()


def main():
    '''This function reproduces some of our results'''

    sys0 = build_system(L=40, W=20, d=0, B=0)
    sys1 = build_system(L=40, W=20, d=20, B=1)
    sys2 = build_system(L=40, W=20, d=20, B=2)


    print('# Plotting system with By=+/-1 dislocations')
    kwant.plot(sys1)

    print('# Plotting SPD for By=+/-1 system')
    plot_spd(sys1)

    print('# Plotting SPD for By=+/-2 system')
    plot_spd(sys2)

    print('# Plot LDOS for doubled H, By=+/-1')
    plot_ldos(sys1, edge_threshold=0.1)

    print('# Plot spectrum of doubled H for By=+/-1')
    plot_doubled_spectrum(sys1)

    print('# Plot localizer index versus origin')
    plot_topind(sys1, originrange=np.linspace(-5.1, 44.9, 51))

    print('# Setting disorder strength=1 and Plotting SPD for By=+/-1 system')
    p['onsdis'] = 1.0
    plot_spd(sys1)
    
    print('# Plotting log(SPD) for the weak Hatano Nelson model')
    p['onsdis'] = 0
    p['PBCx']   = 0
    p['PBCy']   = 0
    plot_spd(sys0)

    print("# Plotting gap and topological index for disordered system")
    plot_gap_index_disorder()

main()

asdasd = input('# Press Enter to exit...')

