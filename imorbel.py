#!/usr/bin/env python

import numpy as np
from multiprocessing import Pool
import corner
from funcs import *

# run from the command line
if __name__ == "__main__":

    # inputs
    parser = argparse.ArgumentParser(description='imorbel - orbital constraints for linear motion')

    parser1 = parser.add_mutually_exclusive_group(required=True)
    parser1.add_argument('--sep',type=float,nargs='+',help='Radial separations')
    parser1.add_argument('--N',type=float,nargs='+',help='North separations')

    parser.add_argument('--e_sep',type=float,nargs='+',help='Radial separation uncertainties')
    parser.add_argument('--e_N',type=float,nargs='+',help='North separation unceratinties')

    parser2 = parser.add_mutually_exclusive_group(required=True)
    parser2.add_argument('--pa',type=float,nargs='+',help='Position angles (E of N)')
    parser2.add_argument('--E',type=float,nargs='+',help='East separations')

    parser.add_argument('--e_pa',type=float,nargs='+',help='Position angle uncertainties')
    parser.add_argument('--e_E',type=float,nargs='+',help='East separation unceratinties')

    parser.add_argument('--date',type=str,nargs='+',help='Dates (YYYY-MM-DD)',required=True)

    parser.add_argument('--mass',type=float,help='Stellar mass (Msun)',required=True)
    parser.add_argument('--e_mass',type=float,help='Stellar mass uncertainty (Msun)',required=True)

    parser.add_argument('--distance',type=float,help='Distance (pc)',required=True)
    parser.add_argument('--e_distance',type=float,help='Distance uncertainty (pc)',required=True)

    parser.add_argument('--nzvz',type=int,help='Number of z/vz grid points',default=100)
    parser.add_argument('--nelem',type=int,help='Number of element samples (approx)',default=10000)

    parser.add_argument('--Qmax',type=float,help='Max apocenter to plot',default=1000.0)

    parser.add_argument('--zvzfile',type=str,help='zvz file name',default='zvz.png')
    parser.add_argument('--elemfile',type=str,help='elem file name',default='elem.png')

    parser.add_argument('--velfit_sky',type=str,help='velfit sky file name',default='velfit_sky.png')
    parser.add_argument('--velfit_tri',type=str,help='velfit corner file name',default='velfit_tri.png')
    parser.add_argument('--velfit_chain',type=str,help='chains file name',default='velfit_chain.png')
    parser.add_argument('--inparams_tri',type=str,help='parameters file name',default='inparams_tri.png')

    parser.add_argument('--interactive','-i',action='store_true',help='Interactive plot')

    args = parser.parse_args()

    # convert lists of floats to numpy ndarrays
    args.sep = np.array(args.sep,dtype=float)
    args.e_sep = np.array(args.e_sep,dtype=float)
    args.pa = np.array(args.pa,dtype=float)
    args.e_pa = np.array(args.e_pa,dtype=float)
    args.N = np.array(args.N,dtype=float)
    args.e_N = np.array(args.e_N,dtype=float)
    args.E = np.array(args.E,dtype=float)
    args.e_E = np.array(args.e_E,dtype=float)
    
    # convert so we can do things in cartesians
    if len(args.sep) > 0:
        N,e_N,E,e_E = seppa2cart(args.sep,args.e_sep,args.pa*np.pi/180.,args.e_pa*np.pi/180.)
    else:
        N,e_N,E,e_E = args.N,args.e_N,args.E,args.e_E

#    print(N,e_N,E,e_E)
        
    # convert the dates
    t = Time(args.date).decimalyear
        
    # get samples of x_0, y_0, V and direction and best fit
    nwalk = 32
    samples = velfit(t,N,e_N,E,e_E,
                     nwalkers=nwalk,nruns=1000,
                     plotsky=True,skyfile=args.velfit_sky,
                     plottri=True,trifile=args.velfit_tri,
                     plotchain=True,chainfile=args.velfit_chain)
    best = np.median(samples,axis=0)

    # extract the best fit parameters
    R = np.sqrt(best[0]**2 + best[1]**2) * args.distance
    pa0 = np.arctan2(best[1],best[0])-np.pi/2. # from E of N, as in samples
    V = best[2] * args.distance
    B = V**2 * R / (8*np.pi**2 * args.mass)
    phi = best[3] - pa0
    if phi > np.pi:
        phi -= 2*np.pi
    zsgn = 1
    if phi < 0:
        zsgn = -1
        phi = -1. * phi

    print('')
    print('R =', R, 'au')
    print('V =', V, 'au / yr')
    print('B =', B)
    print('phi =', phi/np.pi*180, 'deg')
    print('PA0 =', pa0/np.pi*180, 'deg')
    print("Zsign =",zsgn)
    print('')

    titlestr = str(args)+'\n\n'

    # use these contour levels
    contour_levels = default_contour_levels()

    # zvz plot for best fit
    z_vz_data = get_z_vz_data(R,V,B,args.nzvz,args.nzvz)

    # Cycle through z, vz values, and derive orbital elements at each set of values
    element_matrices = get_element_grids(z_vz_data,R,V,B,phi)

    # do interactive plot for best fit and exit when window is closed
    if args.interactive:
        plt.close('all')
        interactive_contour_plot(z_vz_data, element_matrices, contour_levels,R,V,B,phi,pa0,zsgn)
        exit

    # or make plots
    else:

        # z/vz contour plots for best fit
        make_contour_plots(z_vz_data, element_matrices, contour_levels, args.zvzfile, titlestr)

        # get distributions of M, d, B
        Mdist = np.random.normal(loc=args.mass,scale=args.e_mass,size=len(samples))
        ddist = np.random.normal(loc=args.distance,scale=args.e_distance,size=len(samples))
        Rdist = np.sqrt(samples[:,0]**2 + samples[:,1]**2) * ddist
        Vdist = samples[:,2] * ddist
        Bdist = Vdist**2 * Rdist / (8*np.pi**2 * Mdist)
        pa0dist = np.arctan2(samples[:,1],samples[:,0])-np.pi/2. # from E of N

        # ensure phi is in the correct range of -pi..pi
        phidist = samples[:,3] - pa0dist
        while np.min(phidist) <= -np.pi:
            phidist[(phidist <= -np.pi)] += 2.*np.pi
        while np.max(phidist) >= np.pi:
            phidist[(phidist >= np.pi)] -= 2.*np.pi

        # sort out angles, zsgn is -ve when orbit would need to go clockwise (so it is
        # instead viewed from the other size of the sky plane) and is then viewed from
        # -ve z (so phi -> 360-phi)
        zsgn = np.ones(len(samples))
        if np.any(phidist < 0.):
            neg = (phidist < 0.)
            zsgn[neg] = -1.
            phidist[neg] = -1 * phidist[neg]
        phidist *= 180./np.pi
        pa0dist *= 180./np.pi

        # plot the samples from the fit to the astrometry
        extra = np.column_stack((Rdist,Vdist,pa0dist,Bdist,phidist,Mdist,ddist))
        labels = ('$R/au$','$V/au/yr$','$PA_V/^\circ$',
                  '$B$','$\phi/^\circ$',
                  '$M_\star/M_\odot$','$d/pc$',)
        fig = corner.corner(extra,labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True)
        fig.savefig(args.inparams_tri)
        plt.close(fig)

        # Define tested region of z, vz space, maximising by choosing the largest B and
        # smallest R and V (c.f. Pearce+15 eq 4) we will throw away more samples, but
        # won't miss them this way
        z_vz_data = get_z_vz_data(np.min(R),np.min(V),np.max(B),args.nzvz,args.nzvz)

        # compute larger random sample, repeatedly sample from samples, but additional
        # randomisation comes from random sampling of distance and mass
        j = np.random.randint(len(samples),size=args.nelem)
        ddist = np.random.normal(loc=args.distance,scale=args.e_distance,size=args.nelem)
        Mdist = np.random.normal(loc=args.mass,scale=args.e_mass,size=args.nelem)
        Rdist = np.sqrt(samples[j,0]**2 + samples[j,1]**2) * ddist
        pa0dist = np.arctan2(samples[j,1],samples[j,0]) - np.pi/2. # from E of N
        Vdist = samples[j,2] * ddist
        Bdist = Vdist**2 * Rdist / (8*np.pi**2 * Mdist)
        phidist = samples[j,3] - pa0dist

        # same as above
        while np.min(phidist) <= -np.pi:
            phidist[(phidist <= -np.pi)] += 2.*np.pi
        while np.max(phidist) >= np.pi:
            phidist[(phidist >= np.pi)] -= 2.*np.pi
        zsgn = np.ones(args.nelem)
        if np.any(phidist < 0.):
            neg = (phidist < 0.)
            zsgn[neg] = -1.
            phidist[neg] = -1 * phidist[neg]

        # z and vz arrays
        z = z_vz_data['z_list'][np.random.randint(args.nzvz,size=args.nelem)]
        vz = z_vz_data['vz_list'][np.random.randint(args.nzvz,size=args.nelem)]

        # grab a series of random elements, these are correlated so need to select the
        # same i from each
        pars = []
        pars = [np.append(pars,[z[i],vz[i],Rdist[i],Vdist[i],Bdist[i],phidist[i]]) for i in range(args.nelem)]
                             
        pool = Pool(processes=8)
        el = pool.map(calc_elements_array,pars)
        el = np.array(el)

        # select a subset of these, pericenter must be +ve and e<1
        ok = (el[:,1] > 0) & (el[:,3] < 1)
        # and apocenter < Qmax
        ok1 = (el[:,2] < args.Qmax)

        ok = np.all([ok,ok1],axis=0)

        # make corner plot, size pasted from corner code
        K = len(el[0])
        factor = 2.0           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.05         # w/hspace size
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim

        fig,axes = plt.subplots(len(el[0]),len(el[0]),figsize=(dim,dim))
        fig = corner.corner(el[ok],color='k',top_ticks=True,bins=50,fig=fig,
                            labels=('$a/au$','$q/au$','$Q/au$',
                                    '$e$','$I/^\circ$','$\Omega/^\circ$',
                                    '$\omega/^\circ$',r'$\varpi/^\circ$','$f/^\circ$'),
                            range=[1.,(0.,np.max(el[:,1])),1.,(0,1),(0,90),(0,360),(0,360),(0,360),(0,360)])
        axes[1,4].set_title(titlestr)
        fig.savefig(args.elemfile)
        plt.close(fig)
