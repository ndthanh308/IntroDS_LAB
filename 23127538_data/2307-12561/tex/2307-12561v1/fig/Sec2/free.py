import numpy as np
from matplotlib import pyplot as plt
import mpmath as mp
mp.mp.dps = 100

### initial 1 ##

flavor=0
# flavor=int(input('input the initial left chiral state flavor,\n\
# active neutrino: 0\n\
# sterile neutrino: 1\n\
# flavor='))


num1=1001 #number of point

tmax=0.20 #ps

L=np.linspace(0,tmax,num1) #ps

t=L/6.58*1e4 #1/eV

p=mp.mpf('1e-2')#ev

h=-mp.mpf('1')

### initial 1 end ##

def para():
        '''
        #theta12
        #mixing
        theta=33.41*np.pi/180
        #eta=0#np.pi/4

        #mass
        Dm=mp.mpf('7.41e-5')#eV^2
        m0=mp.mpf('1e-3')#eV
        '''
        #seesaw
        m0=mp.mpf('1e-2')#eV
        M=mp.mpf('1')#eV

        theta=np.arcsin(np.sqrt(float(m0/M)))
        Dm=M**2-m0**2
        
        return theta,Dm,m0

def mixing(theta,eta):
	matrix1=np.mat([
		[np.cos(theta),np.sin(theta)],
		[-np.sin(theta),np.cos(theta)]
		])
	
	matrix2=np.mat([
		[1,0],
		[0,np.exp(1j*eta)]
		])
	
	return matrix1*matrix2

def mass(Dm,m0):
	m1=m0#m one, not m L
	m2=np.sqrt(Dm+m0**2)
	return np.array([m1,m2])


def Evo_eq(E,m,h,t0):
	return [[mp.cos(E[0]*t0)+1j*h*p/E[0]*mp.sin(E[0]*t0),-1j*m[0]/E[0]*mp.sin(E[0]*t0),mp.mpf('0'),mp.mpf('0')],
			[-1j*m[0]/E[0]*mp.sin(E[0]*t0),mp.cos(E[0]*t0)-1j*h*p/E[0]*mp.sin(E[0]*t0),mp.mpf('0'),mp.mpf('0')],
			[mp.mpf('0'),mp.mpf('0'),mp.cos(E[1]*t0)+1j*h*p/E[1]*mp.sin(E[1]*t0),-1j*m[1]/E[1]*mp.sin(E[1]*t0)],
			[mp.mpf('0'),mp.mpf('0'),-1j*m[1]/E[1]*mp.sin(E[1]*t0),mp.cos(E[1]*t0)-1j*h*p/E[1]*mp.sin(E[1]*t0)]]


def initial(V):
	#consider a^+ is from \bar{psi}, so initial state mixing has a star
	nu_L = np.kron(V.H,[[1],[0]])
	nu_R = np.kron(V.T,[[0],[1]])
	return nu_L,nu_R#nu_L,nu_L_c,nu_R_c,nu_R

def progress_bar(k):
	print('\r' + str(k*100//(num1-1))+'%', end='', flush=True)

def calculation(flavor,Amplitude,num1,E,h,m,t,nu_L,nu_R,L,save):
	A1,A2,A3,A4=Amplitude
	nu_L=mp.matrix(nu_L)
	nu_R=mp.matrix(nu_R)
	for k in range(0,num1):
		progress_bar(k)
		Evo=mp.matrix(Evo_eq(E,m,h,t[k]))
		A1[k],A2[k]=(nu_L.H*Evo*(nu_L[:,flavor]))  #initial state 
		A3[k],A4[k]=(nu_R.H*Evo*(nu_L[:,flavor]))
	Prob=abs(np.array([A1,A2,A3,A4]))**2
	pic(Prob,L,save)

def pic(P,L,save):
	P1,P2,P3,P4=P

	fig,(ax3,ax4)=plt.subplots(1, 2,figsize=(12, 4))

	ax3.plot(L,P3,'b')
	ax3.set_xlabel(r'$t$ (ps)')
	ax3.set_ylabel('Probability' ) 
	ax3.set_title(r'$\nu_L\rightarrow\nu_{L}^c$')

	ax4.plot(L,P4,'b')
	ax4.set_xlabel(r'$t$ (ps)')
	ax4.set_ylabel('Probability' ) 
	ax4.set_title(r'$\nu_L\rightarrow N_{R}$')

	plt.subplots_adjust(wspace=0.3)

	save()
	plt.show()

def save_free_eta0():
	plt.suptitle(r'$\eta=0$')
	plt.savefig('free_eta0.png',dpi=300)


def save_free_eta45():
	plt.suptitle(r'$\eta=\frac{\pi}{4}$')
	plt.savefig('free_eta45.png',dpi=300)

### initial 2 ##

Amplitude=np.empty((4,num1),dtype=object)

theta,Dm,m0=para()

m=mass(Dm,m0)

E=np.sqrt(p**2+m**2)

h=-mp.mpf('1')


### initial 2 end ##

eta=0#np.pi/4
V=mixing(theta,eta)
nu_L,nu_R=initial(V)

print()
print('free, eta=0')

calculation(flavor,Amplitude,num1,E,h,m,t,nu_L,nu_R,L,save_free_eta0)


eta=np.pi/4
V=mixing(theta,eta)
nu_L,nu_R=initial(V)

print()
print('free, eta=45')

calculation(flavor,Amplitude,num1,E,h,m,t,nu_L,nu_R,L,save_free_eta45)


####

# h=mp.mpf('1')

# print('\n')
# print('free, h=+1')

# calculation(flavor,Amplitude,num1,E,h,m,t,nu_L,nu_R,L,save_free_z)

