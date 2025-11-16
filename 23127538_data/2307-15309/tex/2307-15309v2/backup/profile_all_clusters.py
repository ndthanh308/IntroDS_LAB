# import stuff 
import numpy as np 
import os, sys, glob
import matplotlib.pyplot as plt 
from astropy.cosmology import Planck18_arXiv_v2 as cosmo 
from astropy import units as u


# redshift according to snapshot number
REDSHIFT = {47: 4.99326066076904, 60: 3.50614939996322, 65: 3.02786341652766, 71: 2.53363815515585,
78: 2.02135620134564, 96: 1.03054911183147, 99: 0.899676083658145, 101: 0.817147830286723, 104: 0.700028949584687,
107: 0.592318669581636, 110: 0.48969048645086, 113: 0.39367689886175, 116: 0.303851582652216, 120: 0.193025064610505, 
123: 0.116132168012978, 127: 0.0212616754653474}

# resolution value according to snapshot number 
RES = {47: 0.05, 60: 0.05, 65: 0.05, 71: 0.05, 78: 0.05, 96: 1.25, 99: 1.25, 101: 1.25, 104: 1.25,
107: 1.25, 110: 1.25, 113: 1.25, 116: 1.25, 120: 1.25, 123: 1.25, 127: 1.25}


snapshot = 78
nbins = 5
theta_min = 0
theta_max = 5.0 
theta_edges = np.linspace(theta_min, theta_max, nbins+1)
theta_center = 1/2 * (theta_edges[0:-1] + theta_edges[1:])

dist_min = 0
dist_max = 5 # Mpc
dist_edges = np.linspace(dist_min, dist_max, nbins+1)
dist_center = 1/2 * (dist_edges[0:-1] + dist_edges[1:])


angle_bins = True # True: angular dist (arcmin) vs. False: physical dist (Mpc)
do_beam = True # True: gaussian filter on vs. False: gaussian filter off


def mean_and_err(clusterfile, noisefile):
   cluster = np.genfromtxt(clusterfile)
   minus129 = np.mean(np.concatenate((cluster[:128,:], cluster[129:,:]), axis=0), axis = 0)
   #print(cluster[128,:])
   y = np.mean(cluster, axis = 0)
   dim = cluster.shape
   rows = dim[0]
   noise = np.genfromtxt(noisefile)
   var = (np.var(noise, axis = 0) + np.var(y)) / rows
   std = np.sqrt(var)
   return minus129, y, std


def signal_noise(clusterfile, noisefile):
   cluster = np.genfromtxt(clusterfile)
   noise = np.genfromtxt(noisefile)
   cov = np.cov(np.transpose(noise))
   signal = np.zeros(cluster.shape[0])
   for i in range(cluster.shape[0]):
      profile_signal = cluster[i, :] # ith row
      signal_noise = np.dot(np.dot(np.transpose(profile_signal), np.linalg.inv(cov)), profile_signal) # matrix calculation 
      signal[i] = signal_noise
   return signal

cluster_files = np.sort(glob.glob('cluster{}_*.txt'.format(str(snapshot).zfill(3))))
#print(cluster_files)
noise_angle_files = np.sort(glob.glob('noise_*.txt'))
#print(noise_angle_files)
noise_dist_files = np.sort(glob.glob('noise{}_*.txt'.format(str(snapshot).zfill(3))))
#print(noise_dist_files)

signal = signal_noise(cluster_files[0], noise_dist_files[0])
# cluster - 0: angle False, gs False
# cluster - 1: angle False, gs True
# cluster - 2: angle True, gs False
# cluster - 3: angle True, gs True
# noise_dist_files:
# noise - 0: s4deep, gs False, angle False
# noise - 1: s4deep, gs True, angle False
# noise - 2: s4wide, gs False, angle False
# noise - 3: s4wide, gs True, angle False
# noise_angle_files:
# noise - 0: s4deep, gs False, angle True
# noise - 1: s4deep, gs True, angle True
# noise - 2: s4wide, gs False, angle True
# noise - 3: s4wide, gs True, angle True


"""for sig, num in zip(signal, np.arange(1, 325)):
   if num == 129:
      pass
   else:
      print(num, ':', sig)"""
plt.hist(signal, bins = 100)
plt.yscale('log')
plt.ylabel('SNR in Log Scale')
plt.title(f'z={REDSHIFT[snapshot]: .2f} / bins = 100')
plt.savefig(f'cluster_hist_{snapshot}.png')
plt.show()

#fig, ax = plt.subplots(1, 2)
#fig.set_facecolor('white')
#plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

cl1_exc129, cluster1, s4deep1 = mean_and_err(cluster_files[0], noise_dist_files[0]) # Mpc, gaussian off 
cl1_exc129, cluster1, s4wide1 = mean_and_err(cluster_files[0], noise_dist_files[2]) # Mpc, gaussian off 
cl2_exc129, cluster2, s4deep2 = mean_and_err(cluster_files[1], noise_dist_files[1]) # Mpc, gaussian on 
cl2_exc129, cluster2, s4wide2 = mean_and_err(cluster_files[1], noise_dist_files[3]) # Mpc, gaussian on 

cl3_exc129, cluster3, s4deep3 = mean_and_err(cluster_files[2], noise_angle_files[0]) # arcmin, gaussian off
cl3_exc129, cluster3, s4wide3 = mean_and_err(cluster_files[2], noise_angle_files[2]) # arcmin, gaussian off
cl4_exc129, cluster4, s4deep4 = mean_and_err(cluster_files[3], noise_angle_files[1]) # arcmin, gaussian on 
cl4_exc129, cluster4, s4wide4 = mean_and_err(cluster_files[3], noise_angle_files[3]) # arcmin, gaussian on 

"""plt.errorbar(dist_center, cluster1, yerr = s4deep1, label = 'non-smooth + s4deep')
plt.errorbar(dist_center+0.1, minus129, yerr = s4deep1, label = 'minuus129 + s4deep')
plt.show()"""
#plt.errorbar(dist_center, cluster1, yerr = s4deep1, label = 'non-smooth + s4deep')
#plt.errorbar(dist_center+0.05, cluster1, yerr = s4wide1, label = 'non-smooth + s4wide')
plt.errorbar(dist_center, cluster2, yerr = s4deep2, label = '324 clusters + s4deep')
plt.errorbar(dist_center+0.05, cluster2, yerr = s4wide2, label = '324 clusters + s4wide')
plt.errorbar(dist_center+0.2, cl2_exc129, yerr = s4deep2, label = 'Exc cluster129 + s4deep')
plt.errorbar(dist_center+0.25, cl2_exc129, yerr = s4wide2, label = 'Exc cluster129 + s4wide')
plt.title(f'z ={REDSHIFT[snapshot]: .2f} / Mpc')
plt.ylabel('y')
plt.xlabel('R (Mpc)')
#plt.set_yscale('log')
plt.legend()
plt.savefig(f'cluster_{snapshot}_Mpc_Exc129.png')
plt.show()

plt.errorbar(theta_center, cluster4, yerr = s4deep3, label = '324 clusters + s4deep')
plt.errorbar(theta_center+0.05, cluster4, yerr = s4wide3, label = '324 clusters + s4wide')
plt.errorbar(theta_center+0.2, cl4_exc129, yerr = s4deep4, label = 'Exc cluster129 + s4deep')
plt.errorbar(theta_center+0.25, cl4_exc129, yerr = s4wide4, label = 'Exc cluster129 + s4wide')
plt.title(f'z ={REDSHIFT[snapshot]: .2f} / arcmin')
plt.ylabel('y')
plt.xlabel('theta (arcmin)')
#ax[1].set_yscale('log')
plt.legend()
plt.savefig(f'cluster_{snapshot}_arcmin_Exc129.png')
plt.show()

