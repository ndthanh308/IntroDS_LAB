import matplotlib.pyplot as plt
import numpy as np

u_data = 3
k = 1
alpha = np.concatenate([np.linspace(0,1,10)[:-1],[0.96]])
d = alpha/(1-alpha)

d_i = np.linspace(0,np.amax(d[:-1]),10)
alpha_i = np.interp(d_i, d, alpha)

fixed_points_i = u_data + d_i
fixed_points = u_data + d


X,Y = np.meshgrid(fixed_points, fixed_points_i)

pos = (X-Y >0)*1
neg = (X-Y <0)*-1
yy = (np.sqrt(np.sqrt(pos*(X-Y)))  - np.sqrt(np.sqrt(neg*(X-Y))))
xx = np.zeros_like(yy)

fs = 16
mul = 1.2
plt.figure(figsize=(8,6))
plt.quiver(alpha, fixed_points_i,xx,yy,alpha = 0.8)
plt.plot(alpha_i,fixed_points_i,c='r', label='fixed point')
plt.xlabel('alpha', fontsize=mul*fs)
plt.ylabel('$\mu$_g', fontsize=mul*fs)
plt.yticks(ticks=[3], labels=['$\mu$_data'], fontsize=fs)
plt.xticks(ticks=[0,0.5,1],fontsize=fs)
plt.xlim([0,1])
plt.ylim([2.7,np.amax(d[:-1])])
plt.legend(fontsize=fs, loc = 'upper left')
plt.title('Phase Portrait', fontsize=mul*fs)