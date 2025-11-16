#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

pw = elphmod.bravais.read_pwi('scf.in')

el = elphmod.el.Model('NbS2', rydberg=True)
ph = elphmod.ph.Model('dyn', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('work/NbS2.epmatwp', 'wigner.dat', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(3, 3, shared_memory=True)

driver = elphmod.md.Driver(elph, nk=(16, 16), nq=(1, 1), n=9.0,
    kT=pw['degauss'], f=elphmod.occupations.smearing(pw['smearing']))

driver.kT = 0.005

groups = [(2, [0, 18, 24]), (1, [9, 12, 21]), (1, [3, 6, 15])] # T1
#groups = [(2, [0, 3, 9, 15, 21, 24])] # hexagons
#groups = [(2, [0, 6, 24]), (1, [3, 12, 15]), (1, [9, 18, 21])] # T1'
#groups = [(-2, [0, 6, 24]), (-1, [3, 12, 15]), (-1, [9, 18, 21])] # T2'

for scale, atoms in groups:
    center = np.average(driver.elph.ph.r[atoms], axis=0)

    for atom in atoms:
        u = center - driver.elph.ph.r[atom]
        u *= 0.05 * scale / np.linalg.norm(u)

        driver.u[3 * atom:3 * atom + 3] = u

driver.plot(label=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', tol=1e-8)

driver.plot()

driver.to_xyz('relaxed.xyz')
