set xlabel "# Generations"
set ylabel "Success Rate"
set format y ' %.0f%%'
set key right bottom
plot "figures/success-rate/dominance.dat" u 1:($2*100) w l title "StubCoder (Generation Mode)", \
     "figures/success-rate/repair-mode.dat" u 1:($2*100) w l title "StubCoder (Repair Mode)", \
     "figures/success-rate/weighted-sum.dat" u 1:($2*100) w l title "Weighted Sum", \
     "figures/success-rate/nsga2.dat" u 1:($2*100) w l title "NSGA-II", \
     "figures/success-rate/unguided.dat" u 1:($2*100) w l title "Unguided" \
     #"figures/success-rate/nsga2-mod.dat" u 1:($2*100) w l title "NSGA-II w/ PD", \