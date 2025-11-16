#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

comm = elphmod.MPI.comm

smearings = np.linspace(0.005, 0.020, 16)

def get_nk(sigma):
    return 4 * int(np.ceil(0.02 / sigma)) # 4 originally

pw = elphmod.bravais.read_pwi('scf.in')

el = elphmod.el.Model('NbS2', rydberg=True)
ph = elphmod.ph.Model('dyn', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('work/NbS2.epmatwp', 'wigner.dat', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(3, 3, shared_memory=True)

if comm.rank == 0:
    data = open('free_energy.dat', 'w', buffering=1)

    data.write('%6s %12s %12s\n' % ('sigma', 'sym.', 'CDW'))

nk = 0

for sigma in smearings:
    if nk != get_nk(sigma):
        nk = get_nk(sigma)

        driver = elphmod.md.Driver(elph, nk=(nk, nk), nq=(4, 4), n=9.0,
            kT=pw['degauss'], f=elphmod.occupations.smearing(pw['smearing']))

    driver.kT = sigma
    driver.u[:] = 0.0

    F_sym = driver.free_energy()

    driver.from_xyz('t1.xyz')

    scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
        method='BFGS', tol=1e-8)

    F_CDW = driver.free_energy()

    if comm.rank == 0:
        data.write('%6.4f %12.9f %12.9f\n' % (sigma, F_sym, F_CDW))

    driver.electrons(seedname='NbS2_3x3_%5.3f' % sigma, dk1=4, dk2=4)
    driver.phonons(flfrc='NbS2_3x3_%5.3f.ifc' % sigma)
