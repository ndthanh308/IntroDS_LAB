import numpy as np
import scipy
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix
from matplotlib import rcParams, rc, cm

#final number of psips for rho_N16_H018.3_C100_jz-1.0_mut1.0_mul0.0_b0.txt
Npsip_final = 9893847
Nnorm = 37126
#to store results from the density matrix file
dmdat = []

#reading in the data
with open('rhoz_N16_beta0p5_g0_1p0_h0_0p0_H0_18.3.txt', 'r') as file:
   for row in file:
       a, b, c, d = row.split()
       #a is row, b is column, and c is the density matrix weight for that specific entry in the density matrix
       #d the number of psips for the specific element.
       #taking only the upper triangular part since the density matrix should be symmetric for the Hamiltonian used
       if int(b) >= int(a):           
           dmdat.append([int(a), int(b), float(c), int(d)])
# to sort the list of list using the absolute value of the third element 
# of sublist
def Sort(sub_li):
  
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using absolute value of second element of 
    # sublist lambda has been used
    sub_li.sort(key = lambda x: np.abs(x[2]), reverse = True)
    return sub_li
dmdat = Sort( dmdat )

len_dmdat_full = 0
for i in range(len(dmdat)):
    if dmdat[i][0]==dmdat[i][1]:
        len_dmdat_full += 1
    if dmdat[i][0]!=dmdat[i][1]:
        len_dmdat_full += 2


chi_diag = 0
N_diag = 0

for i in range(len(dmdat)):
    if dmdat[i][0] == dmdat[i][1]:
        chi_diag += dmdat[i][3]


for i in range(len(dmdat)):
    if dmdat[i][0] == dmdat[i][1]:
        N_diag += np.abs(dmdat[i][3])



print('N_diag = ', N_diag)

print('chi_diag = ', chi_diag)




#calculate trace_prime and srhop for each truncated density matrix
trace_value = 0
srhop_check = 0
trace = 0.19533480579647658
srhop = 0.00018380372244269257

range_value = 1000
rho_tilde = []
rho_tilde_errors = []



chi_diag_w = trace*chi_diag
N_diag_w = trace*N_diag



#error propagation
for i in range( range_value ):
    if dmdat[i][1] == dmdat[i][0]:
        trace_value += dmdat[i][2]
        rho_tilde.append((chi_diag/chi_diag_w)*dmdat[i][2])
        rho_tilde_errors.append( (np.sqrt(np.abs(dmdat[i][3]))/(np.abs(chi_diag_w)))*np.sqrt(   1 - 2*dmdat[i][3]/chi_diag_w + (np.abs(dmdat[i][3])/chi_diag_w**2)*N_diag_w   ) )
        #adding contributions to the total error for error propagation, psip number is dmdat[i][3]
    elif dmdat[i][1] > dmdat[i][0]:
        #adding contributions to the total error for error propagation, psip number is dmdat[i][3]
        #since the density matrix is symmetric and only upper triangular half was imported,
        rho_tilde.append((chi_diag/chi_diag_w)*dmdat[i][2])
        rho_tilde.append((chi_diag/chi_diag_w)*dmdat[i][2])
        rho_tilde_errors.append( (np.sqrt(np.abs(dmdat[i][3]))/np.abs(chi_diag_w))*np.sqrt( 1 + (np.abs(dmdat[i][3])/chi_diag_w**2)*N_diag_w   ) )
        rho_tilde_errors.append( (np.sqrt(np.abs(dmdat[i][3]))/np.abs(chi_diag_w))*np.sqrt( 1 + (np.abs(dmdat[i][3])/chi_diag_w**2)*N_diag_w   ) )

#because of off-diagonal elements (each off-diagonal element in the upper triangular half counts twice), only 999 or 1001 are possible for this initial state closest to 1000 elements.
#the total number of density matrix elements for this truncation is 1699.
#the trace of this truncated density matrix is 0.19533480579647658. We then plot the first 1000 elements of this truncated density matrix that has 1699 elements.

x = np.linspace( 1, len(rho_tilde), num=len(rho_tilde), endpoint = True )
dsize = len(x) - 1000
x = list(x)
rho_tilde = list(rho_tilde)
rho_tilde_errors = list(rho_tilde_errors)

del x[-dsize:]
del rho_tilde[-dsize:]
del rho_tilde_errors[-dsize:]

x = np.array(x)
rho_tilde = np.array(rho_tilde)
rho_tilde_errors = np.array(rho_tilde_errors)

plt.rcParams['text.usetex'] = True
plt.rcParams.update(plt.rcParamsDefault)
rc('font',**{'family':'sans-serif', 'size' : 5.5})
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs, amssymb}'
fig,axs = plt.subplots(1, 1, figsize=(3.375, 2.086))
ax1 = axs
plt.xscale('log')
ax1.fill_between(x, rho_tilde - rho_tilde_errors, rho_tilde + rho_tilde_errors, alpha=1.0, color = 'lightgreen')
ax1.plot(x, rho_tilde,  label = r'$\left \langle m | \tilde{\rho}^w | n \right \rangle$', color = 'green', marker='o', markersize=3, linewidth = 0.7) 
plt.axvline(x=50, color = 'red')
ax1.set_xlabel(r'# of $\tilde{\rho}^w \  elements$', fontsize = '11')
ax1.xaxis.set_tick_params(labelsize=11)
ax1.yaxis.set_tick_params(labelsize=11)
ax1.legend(loc="upper right")
plt.savefig('dm_Obs.png', format='png', dpi = 300, bbox_inches='tight', pad_inches = 0.01)
plt.show()

print(trace_value)
print(srhop_check)
