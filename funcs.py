
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
def calc_elements(z,vz,R,V,B,phi,array=0):
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
    if i == 0: O, theta = 0., 0.        # Theta = 0 because r = x
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

    if array:
        return [a,q,Q,e,i,O,w,l,f]

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
def get_element_grids(z_vz_data,R,V,B,phi):
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
            elements = calc_elements(z,vz,R,V,B,phi)

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
    return {'a': [50,120,200,500], \
            'e': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95], \
            'i': [0,15,30,45,60,75,90], \
            'O': [0,45,90,135,180,225,270,315], \
            'w': [0,45,90,135,180,225,270,315], \
            'f': [0,45,90,135,180,225,270,315], \
            'q': [20,50,100,250], \
            'Q': [50,150,250,500,1000], \
            'l': [0,45,90,135,180,225,270,315]}

#------------------------------------------------------------------------------
def make_contour_plots(z_vz_data, element_matrices, contour_levels):
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

    # Display figure
    plt.savefig('zdz.pdf')
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
        el = calc_elements(event.xdata,event.ydata,self.R,self.V,self.B,self.phi)
        [txt.remove() for txt in self.ax.texts]
        self.ax.text(.025,.975,
                     '$a$: {:5.1f}\n$e$: {:4.2f}\n$i$: {:4.1f}\n$\Omega$: {:5.1f}\n$\omega$: {:5.1f}\n$f$: {:5.1f}'.format(el['a'],el['e'],el['i'],el['O'],el['w'],el['f']),
                     transform=self.ax.transAxes, ha='left', \
                     va='top', fontsize = 10, fontname="Times New Roman", \
                     bbox=dict(facecolor='white', edgecolor='white', pad=1), zorder=4)
        if el['e'] > 1.: return
        # orbit in frame where planet lies along x-axis
        f = np.arange(100)/99.*360.
        r = el['a']*(1-el['e']**2)/(1+el['e']*np.cos(f*np.pi/180.))
        x = r * ( np.cos(el['O']*np.pi/180.) * np.cos((el['w']+f)*np.pi/180.) -
                  np.sin(el['O']*np.pi/180.) * np.sin((el['w']+f)*np.pi/180.) * np.cos(el['i']*np.pi/180.) )
        # mirror in y if we're looking from negative z
        y = self.zsgn * r * ( np.sin(el['O']*np.pi/180.) * np.cos((el['w']+f)*np.pi/180.) +
                              np.cos(el['O']*np.pi/180.) * np.sin((el['w']+f)*np.pi/180.) * np.cos(el['i']*np.pi/180.) )
        # convert to real world frame
        r,t = cart2pol(x,y)
        t += np.pi/2. + self.pa0
        x,y = pol2cart(r,t)
        # plot a line if mouse press, otherwise just update
        if event.button != None:
            plt.plot(np.append([0],x),np.append([0],y))
            print('aeiOwf:',el['a'],el['e'],el['i'],el['O'],el['w'],el['f'])
        else:
            self.orb.set_data(np.append([0],x),np.append([0],y))
        self.orb.figure.canvas.draw()

#------------------------------------------------------------------------------
def interactive_contour_plot(z_vz_data, element_matrices, contour_levels,R,V,B,phi,pa0,zsgn):

    # Initialise figure
    fig = plt.figure(figsize=(20,10))
    
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
