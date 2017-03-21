import argparse

import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
from astropy.time import Time

from funcs import *

def orbit(par,t,N,e_N,E,e_E,mstar,distance):
    """Compute residuals for given orbital parameters."""

#    print("par:",par)
    a,e,i,O,w,f = par

    dt = ( t - np.min(t) )
    x_mod = np.array([])
    y_mod = np.array([])
    for j in range(len(N)):
        x_tmp,y_tmp = pos_at_epoch_one(a,e,i,O,w,f,mstar,dt[j])
        x_mod = np.append(x_mod,x_tmp)
        y_mod = np.append(y_mod,y_tmp)

    return x_mod/distance,y_mod/distance


def lnlike(par,t,N,e_N,E,e_E,mstar,distance):
    
    a,e,i,O,w,f = par
    if ( a < 0. or a > 200. or e < 0. or e > 1.
        or i < 0 or i > np.pi
        or O < 0. or O > 2*np.pi or w < 0 or w > 2*np.pi
        or f < 0 or f > 2*np.pi ):
        return -np.inf
        
    x_mod,y_mod = orbit(par,t,N,e_N,E,e_E,mstar,distance)

    resid = np.sqrt( ((E - x_mod)/e_E)**2 + ((N - y_mod)/e_N)**2 )

    return -0.5 * np.sum( resid )


def plotchain(chain,fname):
    nwalkers,nrun,ndim = chain.shape
    plt.figure()
    fig,ax = plt.subplots(ndim,sharex=True)
    fig.subplots_adjust(hspace=0)
    for i in range(ndim):
        for j in range(nwalkers):
            ax[i].plot(chain[j,:,i])
    plt.savefig(fname)
    plt.close(fig)


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
    
    parser.add_argument('--distance',type=float,help='Distance (pc)',required=True)

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

    # convert the dates
    t = Time(args.date).decimalyear

    print(t,N,e_N,E,e_E,args.mass,args.distance)

    ndim = 6
    nwalk = 20
    nrun = 300
    pos = ()
    for j in range(nwalk):
        pos += ([np.random.uniform()*50+5,
                np.random.uniform(),
                np.random.uniform() * np.pi,
                np.random.uniform() * 2*np.pi,
                np.random.uniform() * 2*np.pi,
                np.random.uniform() * 2*np.pi],)

    ini = [ 12.5, 0.01,
           -55.6*np.pi/180., -37.7*np.pi/180.,
           357*np.pi/180., 152*np.pi/180. ]
    pos = [ini + 0.01*np.random.randn(ndim) for i in range(nwalk)]

#    print(pos)

    sampler = emcee.EnsembleSampler(nwalk,ndim,lnlike,threads=4,a=2.0,
                                    args=(t,N,e_N,E,e_E,args.mass,args.distance))
    sampler.run_mcmc(pos,nrun)

    best = sampler.chain[np.unravel_index(np.argmax(sampler.lnprobability),
                                      [nwalk,len(sampler.chain[0,:,0])])]
    print(best)

    plotchain(sampler.chain,'chains.png')
    fig = corner.corner(sampler.chain[:,200:,:].reshape((nrun-200)*nwalk,ndim))
    fig.savefig('corner.png')
    plt.close(fig)

    plt.scatter(E,N)
    xm,ym=orbit(best,t,N,e_N,E,e_E,args.mass,args.distance)
    plt.plot(xm,ym,'r')
    xm,ym=orbit(ini,t,N,e_N,E,e_E,args.mass,args.distance)
    plt.plot(xm,ym,'b')
    for i in np.random.randint(1000,size=50):
        xm,ym = orbit(sampler.flatchain[-i,:],
                      t,N,e_N,E,e_E,args.mass,args.distance)
        plt.plot(xm,ym,alpha=0.2)
    plt.savefig('samp.png')
    plt.close('all')
                 
    print(sampler.acceptance_fraction)
    print(sampler.get_autocorr_time())
