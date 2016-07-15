import math
import corner
import sys
sys.path.append('./')
from funcs import *

# HR 3549B
N1 = -0.806
E1 = -0.333
N2 = -0.776
E2 = -0.348
dt = 3.
d = 92.5
M = 2.35
R,V,B,phi,pa0,zsgn = cart2rvbphi(N1,E1,N2,E2,M,dt,d)

# GQ Lup b
#S1 = 0.739
#PA1 = 275.1
#S2 = 0.725
#PA2 = 276.7
#dt = 2010.-2000.
#d = 140.
#M = 0.7
#R,V,B,phi,pa0,zsgn = seppa2rvbphi(S1,PA1,S2,PA2,M,dt,d)

# Number of z and vz points in the grid (integers)
N_z = 100
N_vz = 100

# Lists of contour levels for each orbital element. Angles in degrees
contour_levels = default_contour_levels()

print()
print('R =', R, 'au')
print('V =', V, 'au / yr')
print('B =', B)
print('phi =', phi/np.pi*180, 'deg')
print('PA0 =', pa0/np.pi*180, 'deg')
print("Zsign =",zsgn)
print()


# Define tested region of z, vz space
z_vz_data = get_z_vz_data(R,V,B,N_z,N_vz)

# Cycle through z, vz values, and derive orbital elements at each set of values
element_matrices = get_element_grids(z_vz_data,R,V,B,phi)

# Save data if necessary
#save_data(z_vz_data, element_matrices)

# Make contour plots
#make_contour_plots(z_vz_data, element_matrices, contour_levels)
interactive_contour_plot(z_vz_data, element_matrices, contour_levels,R,V,B,phi,pa0,zsgn)

#a=[calc_elements(z_vz_data['z_list'][int(math.floor(np.random.random(1)[0]*N_z))],
#                 z_vz_data['vz_list'][int(math.floor(np.random.random(1)[0]*N_z))],
#                 R,V,B,phi,array=1) for i in range(100000)]
#
#a=[calc_elements(z_vz_data['z_list'][int(math.floor(np.random.random(1)[0]*N_z))],
#                 np.random.randn(1)[0]*0.1+0.42,
#                 R,V,B,phi,array=1) for i in range(100000)]
#
#b = np.array(a)
#ok = (b[:,0] > 75 ) & (b[:,0] < 700) & (b[:,3] < 1) # bound
#fig = corner.corner(b[ok],bins=100,color='k',top_ticks=True,
#                    labels=('$a/au$','$q/au$','$Q/au$',
#                            '$e$','$I/^\circ$','$\Omega/^\circ$',
#                            '$\omega/^\circ$',r'$\varpi/^\circ$','$f/^\circ$'),
#                    range=[1.,1.,1.,(0,1),(0,90),(0,360),(0,360),(0,360),(0,360)])
#fig.show()
#fig.savefig('corner.png')

###############################################################################

