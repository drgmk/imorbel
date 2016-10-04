import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import emcee
import corner
import sys
sys.path.append('./')
from funcs import cart2rvbphi,seppa2cart

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

def plotchain(chain,fname,labels=('par0','par1','par2','par3')):
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
def velfit(t,N,Nerr,E,Eerr,nwalkers=32,nruns=1000):

    # get initial parameter guesses from first and last points
    R,V,B,phi,pa0,zsgn = cart2rvbphi(N[0],E[0],N[-1],E[-1],1.0,t[-1]-t[0],1.0)
    par = np.array([np.mean(E),np.mean(N),V,pa0 + zsgn*phi])
#    print(par)

    # find best fit
    ndim = len(par)
    pos = [par + 1e-4*par*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers,ndim,velfit_func,args=(t,N,Nerr,E,Eerr),threads=1)
    pos,_,_ = sampler.run_mcmc(pos,nruns)
    sampler.reset()
    pos,_,_ = sampler.run_mcmc(pos,nruns)

    # make plots
#    print(sampler.acor)
#    print(sampler.acceptance_fraction)
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    labels = ('$x_0$/arcsec','$y_0$/arcsec','$V$/arcsec/yr','$PA_V/rad$')
    plotchain(sampler.chain,'velfit_chain.png',labels=labels)
    fig = corner.corner(samples,labels=labels)
    fig.savefig('velfit_tri.png')
    plt.close(fig)

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
    ax.errorbar(-1*E,N,xerr=Eerr,yerr=Nerr,)
    fig.savefig('velfit_sky.png')
    plt.close(fig)

    # save samples
    return sampler.chain[:, :, :].reshape((-1, ndim))

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
    elif name == 'HD 206893':
        sep = np.array([260.8,274.9])/1e3
        seperr = np.array([4.03,7.9])/1e3
        pa = np.array([69.4,60.96])
        paerr = np.array([0.23,1.06])
        date = ['2015-10-4','2016-08-08']
        d = 40.67
        derr = 0.43
        M = 1.27
        Merr = 0.1
        N,Nerr,E,Eerr = seppa2cart(sep,seperr,pa*np.pi/180.,paerr*np.pi/180.)
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
