import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

df = pd.read_csv('run_data_new.csv', names=["m", 'n', 'optSig', 'optDC', 'time'])


for it in ['optSig', 'optDC']:
    df[it] = df[it].apply(lambda x : True if x == ' True' else False) 

fdf = df[(df['m'] == 4) & (df['n'] <= 10)]

plt.figure(figsize=(6,2))
ax=plt.gca()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:d}'.format(int(y))))

ax.set_xticks(range(2,11))

plt.ylabel('Time (seconds)')
plt.xlabel('Columns ($4 \\times y$)')

no_opt = fdf[~fdf['optSig'] & ~fdf['optDC']]
opt_Sig = fdf[fdf['optSig'] & ~fdf['optDC']]
opt_DC = fdf[~fdf['optSig'] & fdf['optDC']]
full_opt = fdf[fdf['optSig'] & fdf['optDC']]

plt.plot(no_opt['n'], no_opt['time'], 'k-')
plt.plot(opt_Sig['n'], opt_Sig['time'], 'k:', label='cache signature automata')
plt.plot(opt_DC['n'], opt_DC['time'], 'k--', label='adaptive weights for DD graph')
plt.plot(full_opt['n'], full_opt['time'], 'k-.')
plt.legend()
plt.savefig('./All_Optimizations.pdf', bbox_inches='tight')