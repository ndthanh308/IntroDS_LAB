#!/usr/bin/env python3

import elphmod
import numpy as np

comm = elphmod.MPI.comm

smearings = np.linspace(0.005, 0.020, 16)

ph0 = elphmod.ph.Model('NbS2_3x3_0.015.ifc', apply_asr_simple=True)

w02, u0 = np.linalg.eigh(ph0.D())
w0 = elphmod.ph.sgnsqrt(w02)

if comm.rank == 0:
    data = open('modes.dat', 'w')

for i, sigma in enumerate(smearings):
    ph = elphmod.ph.Model('NbS2_3x3_%5.3f.ifc' % sigma, apply_asr_simple=True)

    D = ph.D()

    if comm.rank != 0:
        continue

    w2, u = np.linalg.eigh(D)
    w = elphmod.ph.sgnsqrt(w2) * 1e3 * elphmod.misc.Ry
    u = u.real

    v = (u[:, np.newaxis, :] * u0[:, 3:9, np.newaxis]).sum(axis=0).real

    weight = (abs(v) ** 2).sum(axis=0)

    w = w[weight > 0.5]
    u = u[:, weight > 0.5]
    v = v[:, weight > 0.5]

    U = (ph.r - ph0.r).ravel()
    U *= np.sqrt(np.repeat(ph.M, 3))
    U /= np.linalg.norm(U) or 1.0

    weight_Higgs = abs((u * U[:, np.newaxis]).sum(axis=0)) ** 2

    nu_Higgs = np.argmax(weight_Higgs)

    data.write('%6.4f %6.4f %d\n' % (sigma, weight_Higgs[nu_Higgs], nu_Higgs))

    for nu in range(6):
        data.write('%6.3f' % w[nu])

        for mu in range(6):
            data.write(' %6.3f' % v[mu, nu])

        data.write('\n')

if comm.rank == 0:
    data.close()
