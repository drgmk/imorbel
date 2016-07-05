import math
import corner
import sys
sys.path.append('./')
from funcs import *

# HR 3549B
#N1 = -0.776+0.01327
#E1 = -0.348+0.03017
#N2 = -0.776
#E2 = -0.348
#dt = 3.
#d = 92.5
#M = 2.35

# GQ Lup b
S1 = 0.739
PA1 = 275.1
S2 = 0.725
PA2 = 276.7
dt = 2010.-2000.
d = 140.
M = 0.7

# Number of z and vz points in the grid (integers)
N_z = 1000
N_vz = 1000

# Save plot data as .txt files? (1 = yes, anything else = no)
toggle_save_data = 0

# Lists of contour levels for each orbital element. Angles in degrees
contour_levels = {'a': [50,120,200,500], \
                  'e': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95], \
                  'i': [0,20,40,60,80,100,120,140,160], \
                  'O': [0,45,90,135,180,225,270], \
                  'w': [0,45,90,135,180,225,270], \
                  'f': [0,45,90,135,180,225,270], \
                  'q': [20,50,100,250], \
                  'Q': [50,150,250,500,1000], \
                  'l': [0,45,90,135,180,225,270]}

# convert observables into useful parameters
#R,V,B,phi = cart2rvbphi(N1,E1,N2,E2,M,dt,d)
R,V,B,phi = seppa2rvbphi(S1,PA1,S2,PA2,M,dt,d)

print
print 'R =', R, 'au'
print 'V =', V, 'au / yr'
print 'B =', B
print 'phi =', phi/np.pi*180, 'deg'
print


# Define tested region of z, vz space
z_vz_data = get_z_vz_data(R,V,B,N_z,N_vz)

# Cycle through z, vz values, and derive orbital elements at each set of values
#element_matrices = get_element_grids(z_vz_data,R,V,B,phi)

# Save data if necessary
if toggle_save_data == 1: save_data(z_vz_data, element_matrices)

# Make contour plots
#make_contour_plots(z_vz_data, element_matrices, contour_levels)

a=[calc_elements(z_vz_data['z_list'][int(math.floor(np.random.random(1)[0]*N_z))],
                 z_vz_data['vz_list'][int(math.floor(np.random.random(1)[0]*N_z))],
                 R,V,B,phi,array=1) for i in range(100000)]

#a=[calc_elements(z_vz_data['z_list'][int(math.floor(np.random.random(1)[0]*N_z))],
#                 np.random.randn(1)[0]*0.03+0.42,
#                 R,V,B,phi,array=1) for i in range(100000)]

b = np.array(a)
ok = (b[:,0] < 1e3) & (b[:,1] < 1) # bound
fig = corner.corner(b[ok],bins=100,color='k',top_ticks=True,
                    labels=('$a/au$','$e$','$I/^\circ$','$\Omega/^\circ$','$\omega/^\circ$','$f/^\circ$',
                            '$q/au$','$Q/au$',r'$\varpi/^\circ$'),
                    range=[1.,(0,1),(0,180),(0,360),(0,360),(0,360),1.,1.,(0,360)])
fig.show()
fig.savefig('corner.png')

print 'Program complete'
print
###############################################################################

