import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib import ticker
import seaborn as sns
from matplotlib import font_manager
from matplotlib.patches import Polygon
import networkx as nx
import time
# read data and get X, Y,

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Times New Roman']})
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text',usetex=True)
# mpl.rcParams['font.sans-serif'] = ['Times New Roman']
fs=28
fs1=32
fs2=36
lw=4
labelpad = -30
# mpl.rcParams['font.sans-serif'] = ['Times New Roman']
N = 200
N1 = 100

alpha = 0.6
dt=0.05
# fontdict = {'family': 'Times New Roman',
# 			'size': 18,
# 			'style': 'normal'} # 'normal', 'italic', 'oblique'


d1 = np.load("./data/seed_10_prob_0.18.npy", allow_pickle=True, encoding="latin1")
d2 = np.load("./data/seed_21_prob_0.62.npy", allow_pickle=True, encoding="latin1")
d3 = np.load("./data/seed_46_prob_0.92.npy", allow_pickle=True, encoding="latin1")

X2 = d2.item()['X']
Y2 = d2.item()['Y']

k_12 = float(d2.item()['k_12'])
prob = d2.item()['prob']

# data1
N0=len(X2)
T=N0*dt
print(N0)
data=np.vstack((X2,Y2[:,2:]))[0:60000:10]
r = np.abs(1 / N * np.sum(np.exp(1j * data[:,:]), axis=1))
r1= np.abs(1 / N1 * np.sum(np.exp(1j * data[:,0:100]), axis=1))
r2= np.abs(1 / N1 * np.sum(np.exp(1j * data[:,100:200]), axis=1))
print(data.shape)

# print('data{} syn to {:.3f},group1 syn to {:.3f}, group2 syn to {:.3f},finally desyn to {:.3f}'.format(i,r, r1, r2, r_c_e))


figure,ax= plt.subplots(2,1,figsize = (18,12),sharex=True)

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.90, top=0.95, hspace=0.0, wspace=0.3)
# right,distance portion of right border
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplot(211)
plt.grid(color='black',linewidth=1,alpha=0.3)
plt.plot(r,linewidth=lw,label=r'$r$')
plt.plot(r1,'--',linewidth=lw,label=r'$r_1$')
plt.plot(r2,':',color='#77428D',linewidth=lw,label=r'$r_2$')
plt.ylabel(r'$\mathrm{Order~Paramters}$',fontsize=fs2,labelpad=labelpad+20)
plt.yticks([1],['1'],fontsize = fs1)
plt.xlim(0,6000)
plt.xticks([])
plt.axhline(0.2,label='0.2',ls='dotted',lw=2.5,color='darkgrey')
plt.legend(bbox_to_anchor=(0.95,0.95),loc=1,borderaxespad=0.5,fontsize=fs1,frameon=True,ncol=2)
plt.text(400,0.1,r'$\textbf{control switched on}$',color=[1,0,0],fontsize = 20)
plt.text(6000-350,0.9,r'$\textbf{(a)}$',color='k',fontsize=fs2)
plt.annotate("",
             xy=(1020, 0.1),
             xytext=(1020, -0.5),
             # xycoords="figure points",
             arrowprops=dict(color="red",shrink=0.05))

# ax=plt.gca()
# ax.spines[:].set_linewidth('3.0')

ax2 = plt.subplot(212)
h = plt.imshow(np.sin(data.T), cmap='summer', aspect='auto')
plt.yticks([0, 200],['0','200'], fontsize=fs1)
plt.xticks([0, 1000,6000],['0','500','3000'], fontsize=fs1)
plt.xlabel('Time', fontsize=fs2, labelpad=labelpad)
plt.ylabel('Neuron Index', fontsize=fs2, labelpad=labelpad-10)

plt.annotate("",
             xy=(1020, 25),
             xytext=(1020, -2),
             # xycoords="figure points",
             arrowprops=dict(color="red",shrink=0.05))
plt.text(6000-350,30,r'$\textbf{(b)}$',color='k',fontsize=fs2)
'''
画colorbar
'''
position=figure.add_axes([ax2.get_position().x1 + 0.02, ax2.get_position().y0, 0.01, 2*ax2.get_position().height])
cb = plt.colorbar(h,cax=position)
cb.set_label('Membrane Potential', fontdict={'size':fs2},labelpad=labelpad)
# cb.ax.set_title(r'$v_i$', fontsize=fs2)
cb.set_ticks([-1, 1],['-1','1'])
cb.ax.tick_params(labelsize=fs1)
plt.clim(-1, 1)

plt.savefig('./data/kuramoto_0513_v0.pdf')
plt.show()


