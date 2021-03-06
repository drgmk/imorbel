###############################################################################
'''Given a companion's sky plane position at two epochs, this program produces
contour plots of the companion's possible orbital elements as functions of its
unknown line of sight (z, vz) coordinates at the first observation epoch. For
further details see Pearce, Wyatt & Kennedy 2015. In this version of the code
the companion's sky plane coordinates are inputted as North and East offsets
from the primary. The user should not have to edit anything beyond line 50.
Note: the longitude of ascending node is defined relative to the primary -
companion separation vector at the first epoch of observation. The default
inputs are from Paul Kalas' 2013 paper on Fomalhaut b, and should reproduce
Figure 2 in Pearce, Wyatt & Kennedy 2015. The code produces a Python plot and,
if toggle_save_data = 1, outputs six .txt files, one file for each orbital
element. These files have the z and vz values along the edges, and a grid of
the corresponding orbital element values.'''
###############################################################################

###############################################################################
# Import libraries
import numpy as np                  # Numerical functions
import matplotlib.pyplot as plt     # Plotting functions
from matplotlib import gridspec     # Subplot gridding
import argparse                     # for use as a command line script
import emcee
import corner                       # corner plots
from astropy.time import Time       # time
###############################################################################

###############################################################################
# Functions

def cart2rvbphi(N1,E1,N2,E2,M,dt,d):
    # Calculate the initial sky plane position and speed (Appendix A)
    R = d * (N1**2 + E1**2)**.5
    V = d * ((N2-N1)**2 + (E2-E1)**2)**.5 / dt
    # Calculate B and phi (Equations 1 and A2)
    B = V**2 * R / (8*np.pi**2 * M)
    phi = np.arccos((N1*N2 + E1*E2 - N1**2 - E1**2) \
                    /((N1**2+E1**2)*((N2-N1)**2+(E2-E1)**2))**.5)
    # original PA and whether observer is along +ve z
    pa0 = np.arctan2(-E1,N1)
    pa1 = np.arctan2(-E2,N2)
    zsgn = -1
    if 0. <= (pa1-pa0) <= np.pi or -2*np.pi <= (pa1-pa0) < -np.pi: zsgn = 1
    return (R,V,B,phi,pa0,zsgn)

def seppa2rvbphi(S1,PA1,S2,PA2,M,dt,d):
    PA1 *= np.pi/180.
    PA2 *= np.pi/180.
    R = S1 * d
    V = d * (S1**2 - 2*S1*S2*np.cos(PA2-PA1) + S2**2)**.5 / dt
    # Calculate B and phi (Equations 1 and A1)
    B = V**2 * R / (8*np.pi**2 * M)
    phi = np.arccos((S2*np.cos(PA2-PA1)-S1) \
                    / (S1**2 - 2*S1*S2*np.cos(PA2-PA1) + S2**2)**.5)
    # original PA and whether observer is along +ve z
    pa0 = PA1
    zsgn = -1
    if 0. <= (PA2-PA1) <= 180. or -360. <= (PA2-PA1) < -180.: zsgn = 1
    return (R,V,B,phi,pa0,zsgn)

''' convert sep/PA to cartesian, PA is assumed E of N, and +ve x is E'''
def seppa2cart(sep,seperr,pa,paerr):
    E = -1. * sep * np.sin(pa)
    N = sep * np.cos(pa)
    paerrmag = sep * paerr # assume small angles
    Eerr = np.sqrt( (seperr*np.sin(-pa))**2 + (paerrmag*np.cos(-pa))**2)
    Nerr = np.sqrt( (seperr*np.cos(-pa))**2 + (paerrmag*np.sin(-pa))**2)
    return (N,Nerr,E,Eerr)

def get_vz_max_z(z,R,V,B):
    '''For a given value of z, where |z| < z_max, find maximum value of |vz|
    resulting in a bound orbit. Derived using v^2 r < 2 mu'''

    vz_max_z = V*(B**-1*(1+(z/R)**2)**-.5 - 1)**.5

    return vz_max_z

#------------------------------------------------------------------------------
def get_z_vz_data(R,V,B,N_z,N_vz):
    '''Calculates the maximum values of z and vz resulting in a bound orbit,
    and gets lists of tested z and vz values'''

    # Calculate max values of |z|, |vz| resulting in a bound orbit (Equation 4)
    z_max = R*(B**-2 - 1.)**.5
    vz_max = V*(B**-1 - 1.)**.5

    # Generate list of tested z and vz values
    z_list = np.linspace(-z_max, z_max, N_z)
    vz_list = np.linspace(-vz_max, vz_max, N_vz)

    # Generate line separating bound and unbound orbits
    bound_z_line = list(z_list[:]) + list(reversed(z_list[:]))
    bound_vz_line = []

    for z_ind in range(2*N_z):
        z = bound_z_line[z_ind]

        vz_max_z = get_vz_max_z(z,R,V,B)

        if z_ind <= N_z: bound_vz_line += [vz_max_z]
        else: bound_vz_line += [-vz_max_z]

    # Return dictionary of lists and values
    z_vz_data = {'z_list': z_list, \
                 'vz_list': vz_list, \
                 'bound_z_line': bound_z_line, \
                 'bound_vz_line': bound_vz_line}

    return z_vz_data

#------------------------------------------------------------------------------
def calc_elements_array(p):
    l = calc_elements(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])
    return np.array([l['a'],l['q'],l['Q'],l['e'],l['i'],l['O'],l['w'],l['l'],l['f']])

def calc_elements(z,vz,R,V,B,phi,pa0,zsgn):
    '''Derives orbital elements from position and velocity using the method of
    Murray and Durmott 1999 (equations 2.126 - 2.139). Dimensionless units are
    used, where rho = z/R, nu = vz/V, ap = a/R and hp = h/(VR). Phi is in
    radians'''

    # Define dimensionless line of sight coordinates
    rho = z / R
    nu = vz / V

    # -------------------------- Calculate elements ---------------------------

    # Semimajor axis
    ap = (.5*((1+rho**2)**-.5 - B*(1+nu**2))**-1)
    a = ap * R
    if a < 0: a = 1.e9    # If unbound, set a high to tidy subplot

    hp = (rho**2 - 2*rho*nu*np.cos(phi) + nu**2 + np.sin(phi)**2)**.5
    hxp = - rho*np.sin(phi)
    hyp = rho*np.cos(phi) - nu

    # Eccentricity
    e = (1 - 2*B*hp**2/ap)**.5

    # Pericenter and apocenter
    q = a * (1-e)
    Q = a * (1+e)

    # Inclination
    i = np.arccos(np.sin(phi) / hp)

    Si = np.sin(i)

    # Theta (w+f) and longitude of ascending node
    if i == 0:
        O, theta = 0., 0.        # Theta = 0 because r = x
    else:

        O = np.arctan2(hxp/(hp*Si), -hyp/(hp*Si))

        theta = np.arctan2(rho/(1+rho**2)**.5/Si, \
            (np.cos(O)*(1+rho**2)**.5)**-1*(1+rho*np.sin(O)*np.cos(i)/Si))

        while theta < 0: theta += 2*np.pi
        while theta >= 2*np.pi: theta -= 2*np.pi

    sgn = np.sign(np.cos(phi) + rho*nu)

    # True anomaly
    f = np.arctan2(ap*(1-e**2)/(hp*e)*(1+nu**2-hp**2/(1+rho**2))**.5*sgn, \
        (ap*(1-e**2)/(1+rho**2)**.5 - 1)/e)

    # Argument of pericentre
    w = theta - f

    # account for zsgn
    O = zsgn * pa0 + O
    if zsgn < 0:
        O = np.pi - O
        i = np.pi - i
        w = w + np.pi

    # Convert angles to degrees, and define to lie between 0 and 360 deg:
    i *= 180./np.pi
    O *= 180./np.pi
    w *= 180./np.pi
    f *= 180./np.pi

    while O < 0: O += 360.
    while O >= 360: O -= 360.
    while f < 0: f += 360.
    while f >= 360: f -= 360.
    while w < 0: w += 360.
    while w >= 360: w -= 360.

    # longitude of pericenter
    l = O + w
    while l >= 360: l -= 360.

    # Add elements to dictionary
    elements = {'a': a, \
                'q': q, \
                'Q': Q, \
                'e': e, \
                'i': i, \
                'O': O, \
                'w': w, \
                'l': l, \
                'f': f}

    return elements

#------------------------------------------------------------------------------
def get_element_grids(z_vz_data,R,V,B,phi,pa0,zsgn):
    '''Cycles through z and vz values. For each combination resulting in a
    bound orbit, calculates the corresponding orbital elements. Outputs grids
    of orbital elements for contour plotting.'''

    print('Calculating elements...')

    # Unpack lists of z and vz values
    z_list = z_vz_data['z_list']
    vz_list = z_vz_data['vz_list']
    N_z = len(z_list)
    N_vz = len(z_list)

    # Initiate element grids, to be filled with orbital elements corresponding
    # to z and vz values
    a_mat = np.zeros((N_vz, N_z))
    e_mat = np.zeros((N_vz, N_z))
    i_mat = np.zeros((N_vz, N_z))
    O_mat = np.zeros((N_vz, N_z))
    w_mat = np.zeros((N_vz, N_z))
    f_mat = np.zeros((N_vz, N_z))
    q_mat = np.zeros((N_vz, N_z))
    Q_mat = np.zeros((N_vz, N_z))
    l_mat = np.zeros((N_vz, N_z))

    # Cycle through z and vz values, and derive orbital elements
    for z_ind in range(len(z_list)):
        z = z_list[z_ind]

        for vz_ind in range(len(vz_list)):
            vz = vz_list[vz_ind]

            # Calculate elements corresponding to these z, vz coordinates
            elements = calc_elements(z,vz,R,V,B,phi,pa0,zsgn)

            # Add elements to matrices
            a_mat[vz_ind][z_ind] = elements['a']
            e_mat[vz_ind][z_ind] = elements['e']
            i_mat[vz_ind][z_ind] = elements['i']
            O_mat[vz_ind][z_ind] = elements['O']
            w_mat[vz_ind][z_ind] = elements['w']
            f_mat[vz_ind][z_ind] = elements['f']
            q_mat[vz_ind][z_ind] = elements['q']
            Q_mat[vz_ind][z_ind] = elements['Q']
            l_mat[vz_ind][z_ind] = elements['l']

    # Output matrices
    element_matrices = {'a': a_mat, \
                        'e': e_mat, \
                        'i': i_mat, \
                        'O': O_mat, \
                        'w': w_mat, \
                        'f': f_mat, \
                        'q': q_mat, \
                        'Q': Q_mat, \
                        'l': l_mat}

    return element_matrices

#------------------------------------------------------------------------------
def convmf(m_in,e_in):
    """Convert mean to true anomaly.
        
    Copy of bound orbit part of GMKs IDL straight copy of Vallado's 
    C++ routine.
    """

    m = m_in % (2. * np.pi)
    numiter = 50
    small = 0.00000001
    if e_in > small:

        ecc = e_in

#       ;; /* ------------  initial guess ------------- */
        if (((m < 0.0) and (m > -np.pi)) or (m > np.pi)):
            e0 = m - ecc
        else:
            e0 = m + ecc

        ktr = 1
        e1  = e0 + (m - e0 + ecc * np.sin(e0)) / (1.0 - ecc * np.cos(e0))
        while ((abs(e1 - e0) > small) and (ktr <= numiter)):
            ktr += 1
            e0 = e1
            e1 = e0 + (m - e0 + ecc * np.sin(e0)) / (1.0 - ecc * np.cos(e0))

#       ;; /* ---------  find true anomaly  ----------- */
        sinv = (np.sqrt(1.0 - ecc * ecc) * np.sin(e1)) / (1.0-ecc * np.cos(e1))
        cosv = (np.cos(e1) - ecc) / (1.0 - ecc * np.cos(e1))
        nu   = np.arctan2( sinv, cosv)
            
    else:
#       ;; /* --------------------- circular --------------------- */
        ktr = 0
        nu  = m
        e0  = m

    return nu


#------------------------------------------------------------------------------
def pos_at_epoch_one(a,e,i,O,w,f,mstar,dt):
    """Compute sky X,Y,Z position at dt given elements and stellar mass.

    Angles are given in radians.
    """

    # mean motion (rad/yr)
    n = 2. * np.pi * np.sqrt(mstar/a**3)
    
    # figure out new f at dt, convert f to M, move M, convert back...
    eta = np.sqrt( (1.+e) / (1.-e) )
    Ea = 2. * np.arctan( np.tan(f/2.) / eta )
    M = Ea - e * np.sin(Ea)
    Mep = M + dt * n
    newf = convmf(Mep,e)

    # coords in orbital plane w.r.t. to pericenter
    r = a * ( 1. - e**2 ) / ( 1. + e * np.cos(newf) )
#    x2d = r * np.cos(newf)
#    y2d = r * np.sin(newf)

    # Equation 2.122, Murray & Dermott
    Om = np.pi/2. + O
    cn = np.cos(Om)
    sn = np.sin(Om)
    cp = np.cos(w+newf)
    sp = np.sin(w+newf)
    ci = np.cos(i)
    si = np.sin(i)
    # this is the rotation
    x = r * ( cn*cp - sn*sp*ci )
    y = r * ( sn*cp + cn*sp*ci )
    z = r * ( sp*si )

    return x,y

#------------------------------------------------------------------------------
def pos_at_epoch(element_matrices,mstar,dt):
    '''Return the sky R and X,Y position at a given time.
        
    Receives the elements, stellar mass, and delta time from current
    epoch.
    
    TODO: vectorise for speed (and neatness), would (ideally) mean doing
    the same for convmf.
    '''

    nvz,nz = element_matrices['a'].shape
    x = np.zeros((nvz,nz))
    y = np.zeros((nvz,nz))
    
    for j in range(nvz):
        for k in range(nz):

            # suck out the elements
            a = element_matrices['a'][j,k]
            # go on if a not large (see calc_elements)
            if a == 1e9:
                x[j,k] = 1e9
                y[j,k] = 1e9
                continue
            e = element_matrices['e'][j,k]
            i = element_matrices['i'][j,k] * np.pi/180.
            O = element_matrices['O'][j,k] * np.pi/180.
            w = element_matrices['w'][j,k] * np.pi/180.
            f = element_matrices['f'][j,k] * np.pi/180.
            
            x[j,k],y[j,k] = pos_at_epoch_one(a,e,i,O,w,f,mstar,dt)
            
    return np.sqrt(x**2 + y**2),x,y


#------------------------------------------------------------------------------
def save_data(z_vz_data, element_matrices):
    '''Saves the element grids as .txt files, with the z and vz values along
    the grid edges. Also save the bound / unbound divide line.'''

    print('Saving data...')

    # Unpack v and vz lists and bounding lines
    z_list = z_vz_data['z_list']
    N_z = len(z_list)
    vz_list = z_vz_data['vz_list']
    bound_z_line = z_vz_data['bound_z_line']
    bound_vz_line = z_vz_data['bound_vz_line']

    # Cycle through all six elements, outputting files for each
    for elmnt_str in ['a','e','i','O','w','f']:

        # Get element grid
        mat = element_matrices[elmnt_str]

		# Flip grid in up / down direction (so y axis is positive at top and
		# negative at bottom of output file)
        flipped_mat = np.flipud(mat)

        # Define output file
        grid_file = file('%s_grid.txt' % elmnt_str, 'w')

        # Write out z values along top of grid
        line = ''
        for z in z_list: line += ' %s' % z
        line += '\n'
        grid_file.write(line)

        # Cycle through z and vz values, adding the corresponding elements to
        # the grid. Also add vz values down the left hand side of the grid
        for vz_ind in range(len(vz_list)):

			# Read out vz_list in reverse order (so y axis is positive at top
			# and negative at bottom of output file)
            vz = vz_list[-1-vz_ind]

            line = '%s' % vz

            for z_ind in range(len(z_list)):
                z = z_list[z_ind]        
            
                # Get element matrix entry and add to line
                elmnt = flipped_mat[vz_ind][z_ind]
                line += ' %s' % elmnt

            line += '\n'
            grid_file.write(line)

        grid_file.close()

    # Write bound/unbound divide file
    bound_line_file = file('bound_line.txt', 'w')
    bound_line_file.write('z (au)    vz (au/yr)\n')

    for ind in range(2*N_z):
        z, vz = bound_z_line[ind], bound_vz_line[ind]
        bound_line_file.write('%s %s\n' % (z, vz))

    bound_line_file.close()

#------------------------------------------------------------------------------
def make_individual_cntr_plt(fig, gs, elmnt_str, z_vz_data, element_matrices,
        subplot_pars, contour_levels):
    '''For the given orbital element (elmnt_str), construct the corresponding
    subplot of element contours vs z and vz'''

    # Unpack all required values
    z_list = z_vz_data['z_list']
    vz_list = z_vz_data['vz_list']
    bound_z_line = z_vz_data['bound_z_line']
    bound_vz_line = z_vz_data['bound_vz_line']
    mat = element_matrices[elmnt_str]
    subplot_title = subplot_pars[elmnt_str]['title']
    subplot_number = subplot_pars[elmnt_str]['number']

    # Make subplot
    ax = fig.add_subplot(gs[subplot_number])

    # add filled contours in background
#    if elmnt_str == 'a' or elmnt_str == 'q' or elmnt_str == 'Q':
#        fill_cont = 10**np.arange(np.log10(np.min(contour_levels[elmnt_str])),
#                                  np.log10(np.max(contour_levels[elmnt_str])),
#                                  0.01)
#    else:
#        fill_cont = np.arange(0,np.max(contour_levels[elmnt_str]),
#                              np.max(contour_levels[elmnt_str])/100.)
#
#    CS1 = plt.contourf(z_list, vz_list, mat, levels=fill_cont,alpha=0.3)

    # Plot and label contours, with appropriate numbers of decimal places
    CS = plt.contour(z_list, vz_list, mat, contour_levels[elmnt_str])
    if elmnt_str == 'e':
        ax.clabel(CS, inline=1, fontsize=10, fmt = '%0.1f')
    else: ax.clabel(CS, inline=1, fontsize=10, fmt = '%1.0f')

    # Add bound / unbound line
    ax.plot(bound_z_line, bound_vz_line, 'k--')

    # Set subplot axis labels, if necessary
    if subplot_number in [0,1,2]:
        ax.xaxis.tick_top()
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_label_position('top')

    if subplot_number in [2,5,8]:
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(r'$\dot{z}$ / au yr$^{-1}$', labelpad=20, rotation=270,\
            fontsize = 16, fontname="Times New Roman")

    if subplot_number in [0,3,6]: ax.set_ylabel(r'$\dot{z}$ / au yr$^{-1}$',\
        fontsize = 16, fontname="Times New Roman")

    if subplot_number in [1,2,4,5,7,8]:
        ax.tick_params(labelleft='off')

    if subplot_number in [0,1,2,3,4,5]:
        ax.tick_params(labelbottom='off')
    
    ax.set_xlabel(r'$z$ / au', fontsize = 16, fontname="Times New Roman")

    # Add subplot title
    ax.text(.05,.95, subplot_title, transform=ax.transAxes, ha='left', \
        va='top', fontsize = 20, fontname="Times New Roman", \
        bbox=dict(facecolor='white', edgecolor='white', pad=1), zorder=4)

#------------------------------------------------------------------------------
def default_contour_levels():
    # Lists of contour levels for each orbital element. Angles in degrees
    return {'a': [5,10,20,50,100,200,500], \
            'e': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95], \
            'i': [0,15,30,45,60,75,90,105,120,135,150,165,180], \
            'O': [0,45,90,135,180,225,270,315,360], \
            'w': [0,45,90,135,180,225,270,315,360], \
            'f': [0,45,90,135,180,225,270,315,360], \
            'q': [5,10,15,20,50,100,200,500], \
            'Q': [5,10,20,50,100,200,500,1000], \
            'l': [0,45,90,135,180,225,270,315,360]}

#------------------------------------------------------------------------------
def make_contour_plots(z_vz_data, element_matrices, contour_levels, zvzfile, titlestr):
    '''Plots contours for all six elements as functions of z and vz.'''

    print('Making contour plots...')

    # Initialise figure
    fig = plt.figure(figsize=(12,10))

    # Set subplot ratios and white space widths
    gs = gridspec.GridSpec(3, 3)
    gs.update(hspace=0., wspace = 0.)

    subplot_pars = {'a': {'title': '$a$ / au', 'number': 0}, \
                    'q': {'title': '$q$ / au', 'number': 1}, \
                    'Q': {'title': '$Q$ / au', 'number': 2}, \
                    'e': {'title': '$e$', 'number': 3}, \
                    'i': {'title': '$i$ / $^\circ$', 'number': 4}, \
                    'O': {'title': '$\Omega$ / $^\circ$', 'number': 5}, \
                    'w': {'title': '$\omega$ / $^\circ$', 'number': 6}, \
                    'f': {'title': '$f$ / $^\circ$', 'number': 8}, \
                    'l': {'title': r'$\varpi$ / $^\circ$', 'number': 7}}

    elmnt_strs = ['a', 'q', 'Q', 'e', 'i', 'O', 'w', 'l', 'f']

    # Generate the contour plot for each orbital element
    for elmnt_str in elmnt_strs:
        make_individual_cntr_plt(fig, gs, elmnt_str, z_vz_data, \
            element_matrices, subplot_pars, contour_levels)

    fig.add_subplot(gs[1])
    plt.title(titlestr)
    plt.tight_layout()

    # Display figure
    plt.savefig(zvzfile)
    plt.close()

#------------------------------------------------------------------------------
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#------------------------------------------------------------------------------
class DrawOrbit:
    def __init__(self,orb,ax,R,V,B,phi,pa0,zsgn):
        self.orb = orb
        self.ax = ax
        self.R = R
        self.V = V
        self.B = B
        self.phi = phi
        self.pa0 = pa0
        self.zsgn = zsgn
        self.cid = orb.figure.canvas.mpl_connect('motion_notify_event',self)
        self.cid = orb.figure.canvas.mpl_connect('button_press_event',self)

    def __call__(self,event):
        # only draw if we're inside the plot, and not in the orbit plot
        if event.inaxes == self.orb.axes: return
        if event.xdata == None: return
        el = calc_elements(event.xdata,event.ydata,self.R,self.V,self.B,self.phi,
                           self.pa0,self.zsgn)
        x,y,reali,realom,realw = calc_sky_orbit(el)

        [txt.remove() for txt in self.ax.texts]
        self.ax.text(.025,.975,
                     '$a$: {:5.1f}\n$q$: {:5.1f}\n$Q$: {:5.1f}\n$e$: {:4.2f}\n$i$: {:4.1f}\n$\Omega$: {:5.1f}\n$\omega$: {:5.1f}\n$f$: {:5.1f}'.format(el['a'],el['q'],el['Q'],el['e'],el['i'],el['O'],el['w'],el['f']),
                     transform=self.ax.transAxes, ha='left', \
                     va='top', fontsize = 10, fontname="Times New Roman", \
                     bbox=dict(facecolor='white', edgecolor='white', pad=1), zorder=4)
        if el['e'] > 1.: return

        # plot a line if mouse press, otherwise just update, append [0] so line starts at
        # star and goes to orbit at preicenter (f=0 is first array element)
        if event.button != None:
            plt.plot(np.append([0],x),np.append([0],y))
            print('aeiOwf:',el['a'],el['e'],el['i'],el['O'],el['w'],el['f'])
        else:
            self.orb.set_data(np.append([0],x),np.append([0],y))
        self.orb.figure.canvas.draw()

#------------------------------------------------------------------------------
def calc_sky_orbit(el,pa0=0,zsgn=1):
    """Calculate orbit on the sky.
        
    TODO:merge this with pos_at_epoch
    """

    realom = (pa0*180/np.pi)+zsgn*el['O']
    while realom < 0: realom += 360.
    if zsgn < 0:
        realom += 180.
        reali = 180. - el['i']
        realw = el['w'] + 180.
    else:
        realw = el['w']
        reali = el['i']

    while realom > 360.: realom -= 360.
    while realw > 360.: realw -= 360.
    
    if el['e'] > 1.:
        return 0,0,0,0,0

    fs = np.arange(100)/99.*2*np.pi*0.97
    x = np.zeros(len(fs))
    y = np.zeros(len(fs))
    for j,f in enumerate(fs):
        x[j],y[j] = pos_at_epoch_one(el['a'],el['e'],reali*np.pi/180.,
                                     realom*np.pi/180.,realw*np.pi/180.,f,1.,0.)
    return x,y,reali,realom,realw

    # orbit in frame where planet lies along x-axis
    f = np.arange(100)/99.*350.
    r = el['a']*(1-el['e']**2)/(1+el['e']*np.cos(f*np.pi/180.))
    cosO = np.cos(el['O']*np.pi/180.)
    sinO = np.sin(el['O']*np.pi/180.)
    coswf = np.cos((el['w']+f)*np.pi/180.)
    sinwf = np.sin((el['w']+f)*np.pi/180.)
    cosi = np.cos(el['i']*np.pi/180.)
    x = r * ( cosO * coswf - sinO * sinwf * cosi )
    # mirror in y if we're looking from negative z
    y = r * ( sinO * coswf + cosO * sinwf * cosi ) * zsgn
    # convert to real world frame and add actual ascending node
    r,t = cart2pol(x,y)
    t += np.pi/2. + pa0
    x,y = pol2cart(r,t)
    return x,y,reali,realom,realw


#------------------------------------------------------------------------------
def interactive_contour_plot(z_vz_data, element_matrices, contour_levels,R,V,B,phi,pa0,zsgn):

    # Initialise figure
    fig = plt.figure(figsize=(16,8))
    
    # Set subplot ratios and white space widths
    gs = gridspec.GridSpec(3,3)
    gs.update(hspace=0.,wspace=0.,left=0.04,bottom=0.06,right=0.46,top=0.95)
    
    subplot_pars = {'a': {'title': '$a$ / au', 'number': 0}, \
                    'q': {'title': '$q$ / au', 'number': 1}, \
                    'Q': {'title': '$Q$ / au', 'number': 2}, \
                    'e': {'title': '$e$', 'number': 3}, \
                    'i': {'title': '$i$ / $^\circ$', 'number': 4}, \
                    'O': {'title': '$\Omega$ / $^\circ$', 'number': 5}, \
                    'w': {'title': '$\omega$ / $^\circ$', 'number': 6}, \
                    'f': {'title': '$f$ / $^\circ$', 'number': 7}, \
                    'l': {'title': r'$\varpi$ / $^\circ$', 'number': 8}}

    elmnt_strs = ['a', 'q', 'Q', 'e', 'i', 'O', 'w', 'l', 'f']
        
    # Generate the contour plot for each orbital element
    for elmnt_str in elmnt_strs:
        make_individual_cntr_plt(fig, gs, elmnt_str, z_vz_data, \
                                 element_matrices, subplot_pars, contour_levels)

    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.5,bottom=0.06,right=0.95,top=0.95)

    ax = plt.subplot(gs[0,0])
    ax.axis('equal')
    ax.plot(0,0,'*')
    ax.quiver(R*np.cos(pa0+np.pi/2.),R*np.sin(pa0+np.pi/2.),
              -np.sin(zsgn*phi+pa0),np.cos(zsgn*phi+pa0),angles='xy')
#    ax.quiver(R,0,np.cos(phi),zsgn*np.sin(phi),angles='xy')
    orb, = ax.plot(0,0,linewidth=3)
    ax.set_xlim(np.min(z_vz_data['z_list']),np.max(z_vz_data['z_list']))
    ax.set_ylim(np.min(z_vz_data['z_list']),np.max(z_vz_data['z_list']))
    ax.set_xlabel(r'$x_{sky}$ / au', fontsize = 16, fontname="Times New Roman")
    ax.set_ylabel(r'$y_{sky}$ / au', fontsize = 16, fontname="Times New Roman")
    ax.yaxis.set_label_position('right')
    ax.yaxis.set_ticks_position('right')
    orbit = DrawOrbit(orb,ax,R,V,B,phi,pa0,zsgn)

    plt.show()

###############################################################################
###############################################################################

# functions for fitting sky motion

# function for fitting sky velocity
def velfit_func(par,t,N,Nerr,E,Eerr,verb=0):
    if par[2]<0: return -np.inf
    t0 = np.mean(t)
    # pars are x0,y0, V, phi
    x = par[0] - (t-t0) * par[2] * np.sin(par[3])
    y = par[1] + (t-t0) * par[2] * np.cos(par[3])
    # calculate chi^2
    dist2 = (x-E)**2 + (y-N)**2                          # distance from model to point squared
    ang = np.arctan2(y-N,x-E)                            # at this angle
    err2 = (Eerr*np.cos(ang))**2 + (Nerr*np.sin(ang))**2 # size of error ellipse at ang
    chi2 = dist2/err2
    return -0.5 * np.sum(chi2)

def plot_chain(chain,fname,labels=('par0','par1','par2','par3')):
    nwalkers,nrun,ndim = chain.shape
    plt.figure()
    fig,ax = plt.subplots(ndim,sharex=True)
    fig.subplots_adjust(hspace=0)
    for i in range(ndim):
        ax[i].set_ylabel(labels[i])
        for j in range(nwalkers):
            ax[i].plot(chain[j,:,i])
    ax[-1].set_xlabel('samples')
    plt.savefig(fname)
    plt.close(fig)

'''Return samples from mcmc fitting to companion positions'''
def velfit(t,N,Nerr,E,Eerr,nwalkers=32,nruns=10000,
           plottri=False,trifile='velfit_tri.png',
           plotsky=False,skyfile='velfit_sky.png',
           plotchain=False,chainfile='velfit_chain.png'):

    # get initial parameter guesses from first and last points
    R,V,B,phi,pa0,zsgn = cart2rvbphi(N[0],E[0],N[-1],E[-1],1.0,t[-1]-t[0],1.0)
    par = np.array([np.mean(E),np.mean(N),V,pa0 + zsgn*phi])
#    print(par)

    # find best fit
    ndim = len(par)
    pos = [par + 1e-4*par*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers,ndim,velfit_func,args=(t,N,Nerr,E,Eerr),threads=1)
    pos,_,_ = sampler.run_mcmc(pos,1000)
    sampler.reset()
    pos,_,_ = sampler.run_mcmc(pos,nruns)

    labels = ('$x_0$/arcsec','$y_0$/arcsec','$V$/arcsec/yr','$PA_V/rad$')

    if plotchain:
        plot_chain(sampler.chain,chainfile,labels=labels)

    # array of samples to return, setting angle to 0..360
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    samples[:,3] = samples[:,3] % (2*np.pi) # in case results are crazy large
    while np.min( samples[:,3] ) < 0:
        neg = np.where(samples[:,3] < 0)
        samples[neg,3] = samples[neg,3] + 2*np.pi
    while np.max( samples[:,3] ) > 2*np.pi:
        pos = np.where(samples[:,3] > 0)
        samples[pos,3] -= 2*np.pi

    # sanity check on fit
    ## print(sampler.acor)
    ## print(sampler.acceptance_fraction)

    # make plots
    if plottri:
        fig = corner.corner(samples,labels=labels,
                            quantiles=[0.16, 0.5, 0.84],show_titles=True)
        fig.savefig(trifile)
        plt.close(fig)

    if plotsky:
        fig,ax = plt.subplots()
        ax.axis('equal')
        for i in range(100):
            par1 = sampler.chain[np.random.randint(nwalkers),np.random.randint(nruns),:]
            ax.plot(-1*( par1[0] - (t-np.mean(t)) * par1[2] * np.sin(par1[3]) ),
                    par1[1] + (t-np.mean(t)) * par1[2] * np.cos(par1[3]),alpha=0.1)
        ax.set_xlabel(r'$\alpha$, arcsec')
        ax.set_ylabel(r'$\delta$, arcsec')
        ax.invert_xaxis()
        ax.scatter(-1*E,N)
        ax.errorbar(-1*E,N,xerr=Eerr,yerr=Nerr)
        fig.savefig(skyfile)
        plt.close(fig)

    # return samples
    return samples

def getsys(name):
    if name == 'GQ Lup':
        # from Ginski et al 2014
        datelit = ['1994-04-02','1999-04-10','2002-06-17','2004-06-24','2005-05-25',
                   '2005-08-06','2006-02-20','2006-05-18','2006-07-15','2007-02-16']
        seplit = [0.7138,0.7390,0.7365,0.7347,0.7351,0.7333,0.7298,0.7314,0.7332,0.7300]
        sepliterr = [0.0355,0.011,0.0057,0.0031,0.0033,0.0039,0.0033,0.0035,0.0050,0.0064]
        palit = [275.5,275.62,275.5,275.48,276.00,275.87,276.14,276.06,276.26,276.04]
        paliterr = [1.1,0.86,5,0.25,0.34,0.37,0.35,0.38,0.68,0.63]
        date = ['2008-06-14','2009-06-29','2010-05-05','2011-06-05','2012-03-03']
        sep = [0.7255,0.7264,0.7256,0.7240,0.7240]
        seperr = [0.0050,0.0016,0.0014,0.0020,0.0020]
        pa = [276.66,276.54,276.86,276.94,277.04]
        paerr = [0.50,0.17,0.18,0.23,0.24]
        N,Nerr,E,Eerr = seppa2cart(np.append(seplit,sep),
                                   np.append(sepliterr,seperr),
                                   np.append(palit,pa)*np.pi/180.,
                                   np.append(paliterr,paerr)*np.pi/180.)
        date = np.append(datelit,date)
        d = 140.
        derr = 10.
        M = 0.7
        Merr = 0.00001
    elif name == 'HR3549B':
        N=np.array([-0.806,-0.788,-0.776,-0.775])
        Nerr=np.array([0.009,0.015,0.004,0.009])
        E=np.array([-0.333,-0.334,-0.348,-0.344])
        Eerr=np.array([0.009,0.015,0.004,0.007])
        date=np.array(['2013-01-13','2015-01-14','2015-12-19','2015-12-19'])
        d = 92.5
        derr = 3.
        M = 2.35
        Merr = 0.00001
    elif name == 'PZ Tel':
        datelit = ['2007-06-13','2009-09-27','2010-05-07','2010-10-27','2011-03-25','2011-06-06']
        seplit = [0.2556,0.3366,0.3547,0.3693,0.3822,0.3883]
        sepliterr = [0.0025,0.0012,0.0012,0.0011,0.0010,0.0005]
        palit = [61.68,60.52,60.34,59.91,59.84,59.69]
        paliterr = [0.60,0.22,0.21,0.18,0.19,0.10]
        date = ['2012-06-08','2012-06-08']
        sep = [0.4201,0.4188]
        seperr = [0.0013,0.0014]
        pa = [59.55,59.61]
        paerr = [0.19,0.24]
        N,Nerr,E,Eerr = seppa2cart(np.append(seplit,sep),
                                   np.append(sepliterr,seperr),
                                   np.append(palit,pa)*np.pi/180.,
                                   np.append(paliterr,paerr)*np.pi/180.)
        date = np.append(datelit,date)
        d = 51.5
        derr = 2.5
        M = 1.2
        Merr = 0.00001
    return (Time(date).decimalyear,N,Nerr,E,Eerr,M,Merr,d,derr)
