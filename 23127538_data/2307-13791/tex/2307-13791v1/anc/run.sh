#!/bin/bash

np=8
nk=4

echo 'Using normconserving pseudopotentials from PseudoDojo'
echo '[1] van Setten et al., Comput. Phys. Commun. 226, 39 (2018)'
echo '[2] Hamann, Phys. Rev. B 88, 085117 (2013)'

url=http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard

for pp in Nb.upf S.upf
do
    test -e $pp || (wget $url/$pp.gz && gunzip $pp)
done

mpirun -np $np pw.x -nk $nk < scf.in > scf.out
mpirun -np $np ph.x -nk $nk < ph.in > ph.out

ph2epw

mpirun -np $np pw.x -nk $nk < nscf.in > nscf.out
mpirun -np $np epw.x -nk $np < epw.in > epw.out

mpirun -np $np python3 relax.py
mpirun -np $np python3 rerelax.py

mpirun -np $np python3 dos.py
mpirun -np $np python3 modes.py
mpirun -np $np python3 a2f.py
mpirun -np $np python3 plot.py
