#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines

comm = elphmod.MPI.comm

Margin = 0.9
margin = 0.2

thickness = 0.04

def new_plot(**kwargs):
    plot = storylines.Plot(
        style='NanoLett',

        labelpos='lt',
        labelopt='below right=0.5mm, fill=white, rounded corners',
        labelformat=lambda a: r'\textbf{%s}' % a,

        bottom=Margin,
        margin=margin,

        ymin=0.0,
        ymax=25.0,
        ystep=5.0,
        yminorstep=1.0,
        yminormarks=True,

        **kwargs)

    plot.width = (Margin + 7 * margin - plot.double) / 4
    plot.height = 1.5 * plot.width

    plot.line(grid=True)

    if plot.ylabel is None:
        plot.ylabels = False

    return plot

# panel a:

ph = elphmod.ph.Model('dyn').supercell(3, 3)

if comm.rank == 0:
    r0 = ph.r
    a = ph.a

    for l, title in enumerate(['T1', 'hexagons', "T1$'$", "T2$'$"], 1):
        xyz = '%s.xyz' % title.replace("$'$", 'p').lower()

        atm = np.loadtxt(xyz, skiprows=2, dtype=str, usecols=0)
        r = np.loadtxt(xyz, skiprows=2, dtype=float, usecols=(1, 2, 3))

        plot = storylines.Plot(
            style='NanoLett',
            height=0,
            margin=margin,
            xyaxes=False,
            )

        plot.width = -new_plot().width / 2 + margin

        plot.line(*list(zip(0 * a[0], a[0], a[0] + a[1], a[1], 0 * a[1]))[:2],
            draw='none', fill='yellow!50' if title == 'T1' else 'white')

        tau = np.linalg.norm(r0[1, :2] - r0[0, :2])
        radius = 0.0

        r_sc = np.array([r[i] + m * a[0] + n * a[1]
            for m in [-1, 0, 1]
            for n in [-1, 0, 1]
            for i in range(len(r))])

        bonds = storylines.bonds(R1=r_sc[0::3, :2], R2=r_sc[1::3, :2],
            d1=radius, d2=radius, dmin=0.9 * tau, dmax=1.1 * tau)

        for bond in bonds:
            for bond in storylines.cut(bond, 0, a[1, 1]):
                bond = np.array([elphmod.bravais.rotate(xy, np.pi / 3)
                    for xy in bond])

                for bond in storylines.cut(bond, 0, a[1, 1]):
                    bond = np.array([elphmod.bravais.rotate(xy, -np.pi / 3)
                        for xy in bond])

                    plot.line(*zip(*bond), color='gray')

        u = r - r0

        for i in range(len(u)):
            if i % 3 == 0:
                color = 'black'
            elif i % 3 == 1:
                color = 'gray'
            else:
                continue

            if np.any(abs(u[i, :2]) > 0.033):
                plot.line(*list(zip(r[i], r[i] + 15 * u[i]))[:2],
                    thick=True, color=color,
                    **{'->': True, 'shorten >': '-1mm', 'shorten <': '1mm'})

        plot.line(*r[0::3, :2].T, mark='ball', mark_size='1mm',
            ball_color='gray', only_marks=True)

        plot.line(*r[1::3, :2].T, mark='ball', mark_size='1mm',
            ball_color='yellow', only_marks=True)

        plot.line(*list(zip(0 * a[0], a[0], a[0] + a[1], a[1], 0 * a[1]))[:2])

        plot.node(0.5 * a[1, 1], -0.5 * a[1, 0], title, above=True, rotate=30)

        for line in plot.lines:
            line['x'], line['y'] = line['y'], -np.array(line['x'])

        plot.save('fig_a%d.pdf' % l)

    plot = new_plot(label='a', xyaxes=False)

    plot.lines = []

    plot.width = 2 * margin - plot.width
    plot.height = Margin + margin - plot.height

    plot.xmin = 0.0
    plot.xmax = 4.0
    plot.ymin = 0.0
    plot.ymax = 4.0

    plot.node(1, 2.7, r'\includegraphics{fig_a1}')
    plot.node(3, 2.7, r'\includegraphics{fig_a2}')
    plot.node(1, 0.7, r'\includegraphics{fig_a3}')
    plot.node(3, 0.7, r'\includegraphics{fig_a4}')

    plot.save('fig_a.pdf')

# panel b:

ymin = -0.4
ymax = 1.0
yz = 0.04

el_sym = elphmod.el.Model('NbS2')
el = elphmod.el.Model('NbS2_3x3_0.005')

k_sym, x, corners = elphmod.bravais.path('GMKG', ibrav=4, N=600)
k = 3 * k_sym

e_sym, U_sym = elphmod.dispersion.dispersion(el_sym.H, k_sym, vectors=True)
e, U = elphmod.dispersion.dispersion(el.H, k, vectors=True)

W = elphmod.dispersion.unfolding_weights(k_sym,
    elphmod.bravais.supercell(3, 3)[-1], U_sym, U)

if comm.rank == 0:
    plot = new_plot(
        label='b',

        xticks=list(zip(x[corners], [
            r'$\Gamma$',
            r'$\mathrm M$',
            r'$\mathrm K$',
            r'$\Gamma$',
            ])),

        xlabel='Original Brillouin zone',
        ylabel='Electron energy (eV)',

        left=Margin,
        )

    plot.ymin = ymin
    plot.ymax = ymax
    plot.ystep = 0.5
    plot.yminorstep = 0.1

    for n in range(9):
        plot.compline(x, e[:, n], W[:, n], colors=['cyan', 'black'],
            thickness=2 * thickness, protrusion=10.0, threshold=0.01, cut=True)

    plot.node(x[-1], plot.ymin, r'5\,mRy', fill='white', rounded_corners='1pt',
        above_left=True)

    plot.save('fig_b.pdf')

# panel c:

alphas = np.linspace(0.0, 1.0, 11)

if comm.rank == 0:
    plot = new_plot(
        label='c',

        xmin=0.0,
        xmax=5.5,
        xstep=1.0,
        xminorstep=0.5,
        xminormarks=True,

        xlabel='Density of states (1/eV)',

        lower='cyan',
        upper='gray',

        colorbar=False,
        )

    plot.ymin = ymin
    plot.ymax = ymax
    plot.ystep = 0.5
    plot.yminorstep = 0.1

    for point in -yz, +yz:
        plot.line(y=point, color='lightgray')

    for m, alpha in enumerate(alphas):
        last = m == len(alphas) - 1

        w, DOS = np.loadtxt('dos_%3.1f.dat' % alpha).T

        xref = 2.5 * (1 - alpha)
        plot.line(DOS + xref, w, draw='none', xref=xref if last else plot.ymin,
            fill='cyan!50!white' if last else 'white', cut=True)
        plot.line(DOS + xref, w, xref, xref=xref, cut=True)

    plot.code(r'\fill[lightgray] '
            '(%g, %g) -- (%g, %g) -- (%g, %g) -- (%g, %g) -- cycle;' % (
        -plot.width, (+yz - plot.ymin) / (plot.ymax - plot.ymin) * -plot.height,
        margin + Margin / 2 - plot.width, -plot.height,
        margin + Margin / 2 - plot.width, 0.0,
        -plot.width, (-yz - plot.ymin) / (plot.ymax - plot.ymin) * -plot.height,
        ))

    plot.node(plot.xmax, plot.ymin, r'5\,mRy', fill='white',
        rounded_corners='1pt', above_left=True)

    plot.save('fig_c.pdf')

# panel d:

if comm.rank == 0:
    plot = new_plot(
        label='d',

        xmin=0.0,
        xmax=5.5,
        xstep=1.0,
        xminorstep=0.5,
        xminormarks=True,

        xlabel='Density of states (1/eV)',

        lower='cyan',
        upper='gray',

        colorbar=False,
        )

    plot.ymin = -yz
    plot.ymax = +yz
    plot.yticks = [(-yz, r'\smash{\llap\textminus0\rlap{.04}}'),
        -0.02, 0, 0.02, (0.04, r'\llap{0.0}4')]
    plot.yminorstep = 0.01
    plot.ylabels = True

    plot.left = Margin / 2
    plot.width += Margin / 2 - margin

    for m, alpha in enumerate(alphas):
        last = m == len(alphas) - 1

        w, DOS = np.loadtxt('dos_%3.1f_zoom.dat' % alpha).T

        xref = 2.5 * (1 - alpha)
        plot.line(DOS + xref, w, draw='none', xref=xref if last else plot.ymin,
            fill='cyan!50!white' if last else 'white', cut=True)
        plot.line(DOS + xref, w, xref, xref=xref, cut=True)

    plot.code(r'\fill[lightgray] '
            '(%g, %g) -- (%g, %g) -- (%g, %g) -- (%g, %g) -- cycle;' % (
        -Margin / 2 - margin, (+yz - ymin) / (ymax - ymin) * -plot.height,
        0.0, -plot.height,
        0.0, 0.0,
        -Margin / 2 - margin, (-yz - ymin) / (ymax - ymin) * -plot.height,
        ))

    plot.line(x=0, color='gray')

    plot.save('fig_d.pdf')

# panel e:

smearing = 0.013

ph0 = elphmod.ph.Model('NbS2_3x3_0.015.ifc', apply_asr_simple=True)
ph = elphmod.ph.Model('NbS2_3x3_%5.3f.ifc' % smearing, apply_asr_simple=True)
ph_sym = elphmod.ph.Model('dyn', apply_asr_simple=True)

w02, u0 = np.linalg.eigh(ph0.D())

def weights(u):
    v = np.empty((len(q), ph.size, 2))
    v[:, :, 0] = (abs((u[:, :, :, np.newaxis]
        * u0[np.newaxis, :, np.newaxis, 3:9]).sum(axis=1)) ** 2).sum(axis=2)
    v[:, :, 1] = 1 - v[:, :, 0]
    return v

q, x, corners = elphmod.bravais.path('GMK', ibrav=4, N=300)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
w = elphmod.ph.sgnsqrt(w2) * 1e3 * elphmod.misc.Ry

v = weights(u)

if comm.rank == 0:
    plot = new_plot(
        label='e',

        xticks=list(zip(x[corners], [
            r'$\Gamma$',
            r"$\mathrm M\smash'$",
            r"$\mathrm K\smash'$",
            ])),

        xlabel='Supercell Brillouin zone',
        ylabel=r'Phonon energy (meV)',

        left=Margin,
        )

    for nu in range(ph.size):
        plot.compline(x, w[:, nu], v[:, nu], colors=['magenta', 'black'],
            thickness=thickness, protrusion=10.0, cut=True)

    plot.node(x[-1], 0.0, r'%g\,mRy' % (1e3 * smearing), fill='white',
        rounded_corners='1pt', above_left=True)

    plot.save('fig_e.pdf')

# panel f:

q_sym, x, corners = elphmod.bravais.path('GMK', ibrav=4, N=300)
q = 3 * q_sym

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
w2_sym, u_sym = elphmod.dispersion.dispersion(ph_sym.D, q_sym, vectors=True)
w = elphmod.ph.sgnsqrt(w2) * 1e3 * elphmod.misc.Ry

v = weights(u) * elphmod.dispersion.unfolding_weights(q_sym,
    elphmod.bravais.supercell(3, 3)[-1], u_sym, u)[:, :, np.newaxis]

if comm.rank == 0:
    plot = new_plot(
        label='f',

        xticks=list(zip(x[corners], [
            r'$\Gamma$',
            r'$\mathrm M$',
            r'$\mathrm K$',
            ])),

        xlabel='Original Brillouin zone',
        )

    for nu in range(ph.size):
        plot.compline(x, w[:, nu], v[:, nu], colors=['magenta', 'black'],
            thickness=thickness, protrusion=10.0, threshold=0.01, cut=True)

    plot.node(x[-1], 0.0, r'%g\,mRy' % (1e3 * smearing), fill='white',
        rounded_corners='1pt', above_left=True)

    plot.save('fig_f.pdf')

# panel g:

if comm.rank == 0:
    smearings = np.linspace(0.005, 0.020, 16)

    w = np.empty((len(smearings), 6))
    u = np.empty((len(smearings), 6, 6))

    with open('modes.dat') as lines:
        for line in lines:
            sigma = float(line.split()[0])

            for nu in range(6):
                x = list(map(float, next(lines).split()))
                w[np.argmin(abs(smearings - sigma)), nu] = x[0]
                u[np.argmin(abs(smearings - sigma)), :, nu] = x[1:]

    smearings = np.array(smearings)
    smearings *= 1e3

    plot = new_plot(
        label='g',

        xlabel='Cold smearing (mRy)',
        xmin=2.0,
        xminorstep=1.0,
        xminormarks=True,

        lpos='rt',
        lopt='below left=1mm, draw=gray, fill=white, rounded corners=1pt',
        )

    for point in 5.0, 13.0:
        plot.line(x=point, dash_pattern='on 0.8mm off 0.8mm')

    plot.axes()

    order = elphmod.dispersion.band_order(w, u)

    for i in range(len(smearings)):
        w[i] = w[i, order[i]]

        for mu in range(6):
            u[i, mu] = u[i, mu, order[i]]

    ok = smearings < 14.7

    for nu in range(6):
        plot.line(smearings[ok], w[ok, nu], mark='x', color='magenta', cut=True,
            very_thick=nu == 5)

    ok = smearings > 14.7

    plot.line(smearings[ok], w[ok, 0], mark='*', color='black')

    box = 'fill=white, draw=lightgray, rounded corners=1pt'

    plot.code(r'\draw [<-, thick] (<x=%g>, <y=%g>) -- +(%g, %g) '
        'node [above, %s] {soft mode};' % (17.5, 8.5, -0.3, 0.6, box))

    plot.code(r'\draw [<-, thick] (<x=%g>, <y=%g>) -- +(%g, %g) '
        'node [above, %s] {amplitude mode};' % (8.5, 16.5, 0.6, 0.6, box))

    plot.code(r'\draw [<-, thick] (<x=%g>, <y=%g>) -- +(%g, %g) '
        'node [below, %s] {phase modes};' % (6.5, 4.5, 0.2, -0.3, box))

    plot.save('fig_g.pdf')

# panel h:

if comm.rank == 0:
    plot = new_plot(
        label='h',

        xmin=0.0,
        xmax=0.45,
        xstep=0.1,
        xminorstep=0.05,
        xminormarks=True,

        xlabel='Density of states (1/meV)',

        lpos='rt',
        lopt='below left=1mm',
        lbox=True,
        )

    omega, DOS, a2F = np.loadtxt('a2f.dat').T

    dw = omega[1] - omega[0]

    a2F *= 3 / (a2F[omega <= 25].sum() * dw)
    DOS *= 3 / (DOS[omega <= 25].sum() * dw)

    plot.line(DOS, omega, draw='none', xref=0.0, cut=True, fill='lightgray')

    plot.axes()

    plot.line(a2F, omega, draw='magenta', xref=0.0, cut=True)

    plot.line(label='DOS', color='lightgray', line_width='2mm', line_cap='butt')
    plot.line(label=r'Eliashberg', color='magenta')

    plot.node(plot.xmax, 0.0, r'5\,mRy', fill='white', rounded_corners='1pt',
        above_left=True)

    plot.save('fig_h.pdf')

# combine panels:

if comm.rank == 0:
    storylines.combine('fig.png', ['fig_%s' % a for a in 'abcdefgh'], columns=4)
