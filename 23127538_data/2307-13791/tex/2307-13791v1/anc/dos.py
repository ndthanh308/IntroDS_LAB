#!/usr/bin/env python

import elphmod
import numpy as np

nk = 40 # 400
nw0 = 141 # 1401
nwz = 81 # 801

pw = elphmod.bravais.read_pwi('scf.in')

el_sym = elphmod.el.Model('NbS2', rydberg=True)
ph_sym = elphmod.ph.Model('dyn', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('work/NbS2.epmatwp', 'wigner.dat', el_sym, ph_sym,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(3, 3, shared_memory=True)

driver = elphmod.md.Driver(elph, nk=(16, 16), nq=(1, 1), n=9.0,
    kT=pw['degauss'], f=elphmod.occupations.smearing(pw['smearing']))

driver.kT = 0.005

alphas = np.linspace(0.0, 1.0, 11)

for i, alpha in enumerate(alphas):
    driver.from_xyz('t1.xyz')
    driver.u *= alpha

    driver.diagonalize()

    el = driver.electrons(dk1=4, dk2=4) if alpha else el_sym

    e = elphmod.dispersion.dispersion_full_nosym(el.H, nk, shared_memory=True)

    if not alpha and elphmod.MPI.shm_split()[0].rank == 0:
        e -= driver.mu
        e *= elphmod.misc.Ry

    for wmin, wmax, nw, label in [
        (-0.40, +1.00, nw0, ''),
        (-0.04, +0.04, nwz, '_zoom')]:

        w = np.linspace(wmin, wmax, nw)

        DOS = 0

        status = elphmod.misc.StatusBar(el.size // 3, title='calculate DOS')

        for n in range(el.size // 3):
            DOS += elphmod.dos.hexDOS(e[:, :, n], minimum=wmin, maximum=wmax)(w)

            status.update()

        DOS /= el.size // 3

        if elphmod.MPI.comm.rank == 0:
            with open('dos_%3.1f%s.dat' % (alpha, label), 'w') as data:
                for iw in range(nw):
                    data.write('%7.4f %9.6f\n' % (w[iw], DOS[iw]))
