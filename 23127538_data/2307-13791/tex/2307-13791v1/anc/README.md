# Unconventional charge-density-wave gap in monolayer NbS₂

This directory contains the Quantum ESPRESSO input files and Python scripts
necessary to reproduce Fig. 4 of our paper "Unconventional charge-density-wave
gap in monolayer NbS₂" by Knispel, Berges, Schobert, van Loon, Jolie, Wehling,
Michely, and Fischer (2023).

We have performed all DFT and DFPT calculations using version 7.1 of Quantum
ESPRESSO. Any similarly recent version should work equally well. To interface
the EPW code to our *elphmod* package, the file `EPW/src/ephwann_shuffle.f90`
has to be modified. After the determination of the Wigner-Seitz points, just
before the Bloch-to-Wannier transform, add the following lines:

    IF (ionode) THEN
      OPEN (13, file='wigner.dat', action='write', status='replace', &
        access='stream')
      WRITE (13) dims, dims2
      WRITE (13) nrr_k, irvec_k, ndegen_k
      WRITE (13) nrr_g, irvec_g, ndegen_g
      CLOSE (13)
    ENDIF

Our *elphmod* and *StoryLines* packages and all other Python requirements can be
installed in a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install -r requirements.txt

A LaTeX installation, preferably TeX Live, is required to typeset the figure.

The workflow of the calculation is specified in the Bash script `run.sh`. After
the *ab initio* steps, the following data files should be present:

    NbS2_hr.dat
    NbS2_wsvec.dat
    dyn0 ... dyn19
    wigner.dat
    work/NbS2.epmatwp
