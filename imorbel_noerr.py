#!/usr/bin/env python

import sys
sys.path.append('/Users/grant/code/github/imorbel')
from funcs import *
import numpy as np                  # Numerical functions
from multiprocessing import Pool
import argparse                     # for use as a command line script
import corner                       # corner plots
from astropy.time import Time       # time

# run from the command line
if __name__ == "__main__":

    # inputs
    parser = argparse.ArgumentParser(description='imorbel - orbital constraints for linear motion')

    parser1 = parser.add_mutually_exclusive_group(required=True)
    parser1.add_argument('--sep1',type=float,help='Separation at epoch 1')
    parser1.add_argument('--N1',type=float,help='North separation at epoch 1')

    parser2 = parser.add_mutually_exclusive_group(required=True)
    parser2.add_argument('--sep2',type=float,help='Separation at epoch 2')
    parser2.add_argument('--N2',type=float,help='North separation at epoch 2')

    parser3 = parser.add_mutually_exclusive_group(required=True)
    parser3.add_argument('--E1',type=float,help='East separation at epoch 1')
    parser3.add_argument('--pa1',type=float,help='Position angle (E of N) at epoch 1')

    parser4 = parser.add_mutually_exclusive_group(required=True)
    parser4.add_argument('--E2',type=float,help='East separation at epoch 2')
    parser4.add_argument('--pa2',type=float,help='Position angle (E of N) at epoch 2')

    parser.add_argument('--date1',type=str,help='Date (YYYY-MM-DD) of epoch 1',required=True)
    parser.add_argument('--date2',type=str,help='Date (YYYY-MM-DD) of epoch 2',required=True)

    parser.add_argument('--mass','-m',type=float,help='Stellar mass (Msun)',required=True)
    parser.add_argument('--distance','-d',type=float,help='Distance (pc)',required=True)

    parser.add_argument('--nzvz',type=int,help='Number of z/vz grid points',default=100)

    parser.add_argument('--nelem',type=int,help='Number of grid points to sample',default=10000)
    parser.add_argument('--Qmax',type=float,help='Max apocenter to plot',default=1000.0)

    parser.add_argument('--interactive','-i',action='store_true',help='Interactive plot')

    parser.add_argument('--zvzfile',type=str,help='zvz file name',default='zvz.png')
    parser.add_argument('--elemfile',type=str,help='elem file name',default='elem_noerr.png')

    args = parser.parse_args()

    # dt using astropy Time
    d = Time(args.date2)-Time(args.date1)
    dt = d.value/365.25

    # compute the basic parameters
    if args.sep1 != None:
        R,V,B,phi,pa0,zsgn = seppa2rvbphi(args.sep1,args.pa1,args.sep2,args.pa2,args.mass,dt,args.distance)
        titlestr =  'Input parameters\n$S_1:'+str(args.sep1)+'$"  $PA_1:'+str(args.pa1)+'^\circ$  $D_1$:'+str(args.date1)+'    $S_2:'+str(args.sep2)+'$"  $PA_2:'+str(args.pa2)+'^\circ$  $D_2$:'+str(args.date2)+'    $M_\star:'+str(args.mass)+'M_\odot$  $d:'+str(args.distance)+'pc$\n\n\n'
    else:
        R,V,B,phi,pa0,zsgn = cart2rvbphi(args.N1,args.E1,args.N2,args.E2,args,mass,dt,args.distance)
        titlestr = 'write me!'

    print('')
    print('R =', R, 'au')
    print('V =', V, 'au / yr')
    print('B =', B)
    print('phi =', phi/np.pi*180, 'deg')
    print('PA0 =', pa0/np.pi*180, 'deg')
    print("Zsign =",zsgn)
    print('')

    # use these contour levels
    contour_levels = default_contour_levels()

    # range of z and vz explored
    N_z = args.nzvz
    N_vz = N_z
    z_vz_data = get_z_vz_data(R,V,B,N_z,N_vz)

    # Cycle through z, vz values, and derive orbital elements at each set of values
    element_matrices = get_element_grids(z_vz_data,R,V,B,phi)

    # do interactive plot
    if args.interactive:
        interactive_contour_plot(z_vz_data, element_matrices, contour_levels,R,V,B,phi,pa0,zsgn)

    # or make plots
    else:

        # z/vz contour plots
        make_contour_plots(z_vz_data, element_matrices, contour_levels, args.zvzfile, titlestr)

        # select random points in z/vz space and get elements
        ## a = np.array([calc_elements_array([z_vz_data['z_list'][np.random.randint(N_z)],
        ##                                   z_vz_data['vz_list'][np.random.randint(N_vz)],
        ##                                   R,V,B,phi]) for i in range(args.nelem)])

        # parallel version, slightly faster but not much of a saving as lots of time is
        # spent in get_element_grids above
        z = z_vz_data['z_list'][np.random.randint(N_z,size=args.nelem)]
        vz = z_vz_data['vz_list'][np.random.randint(N_vz,size=args.nelem)]
        pars = []
        pars = [np.append(pars,[z[i],vz[i],R,V,B,phi]) for i in range(args.nelem)]
        pool = Pool(processes=8)
        a = np.array( pool.map(calc_elements_array,pars) )

        # select a subset of these, pericenter must be +ve and e<1
        ok = (a[:,1] > 0) & (a[:,3] < 1)
        # and apocenter < X
        ok1 = (a[:,2] < args.Qmax)

        ok = np.all([ok,ok1],axis=0)

        # make corner plot, size pasted from corner code
        K = len(a[0])
        factor = 2.0           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.05         # w/hspace size
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim

        fig,axes = plt.subplots(len(a[0]),len(a[0]),figsize=(dim,dim))
        fig = corner.corner(a[ok],bins=100,color='k',top_ticks=True,fig=fig,
                            labels=('$a/au$','$q/au$','$Q/au$',
                                    '$e$','$I/^\circ$','$\Omega/^\circ$',
                                    '$\omega/^\circ$',r'$\varpi/^\circ$','$f/^\circ$'),
                            range=[1.,(0.,np.max(a[:,1])),1.,(0,1),(0,90),(0,360),(0,360),(0,360),(0,360)])

        axes[1,4].set_title(titlestr)
        fig.savefig(args.elemfile)

###############################################################################
