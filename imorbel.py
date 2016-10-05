import numpy as np
from multiprocessing import Pool
import corner
from velfit import *
from funcs import get_z_vz_data,calc_elements_array

def elem(t,N,Nerr,E,Eerr,M,Merr,d,derr):

    # samples of x_0, y_0, V and direction and best fit
    samples = velfit(t,N,Nerr,E,Eerr)
    best = np.median(samples,axis=0)

    # best fit parameters
    R = np.sqrt(best[0]**2 + best[1]**2) * d
    pa0 = np.arctan2(best[1],best[0])-np.pi/2. # from E of N, as in samples
    V = best[2] * d
    B = V**2 * R / (8*np.pi**2 * M)

    # distributions of M, d, B
    Mdist = np.random.normal(loc=M,scale=Merr,size=len(samples))
    ddist = np.random.normal(loc=d,scale=derr,size=len(samples))
    Rdist = np.sqrt(samples[:,0]**2 + samples[:,1]**2) * ddist
    Vdist = samples[:,2] * ddist
    Bdist = Vdist**2 * Rdist / (8*np.pi**2 * Mdist)
    pa0dist = np.arctan2(samples[:,1],samples[:,0])-np.pi/2. # from E of N

    zsgn = np.ones(len(samples))
    phidist = samples[:,3] - pa0dist
    phidist += 2.*np.pi
    # ensure phi is defined correctly
    while np.max(phidist) >= np.pi:
        phidist[(phidist >= np.pi)] -= 2.*np.pi
    if np.any(phidist < 0.):
        neg = (phidist < 0.)
        zsgn[neg] = -1.
        phidist[neg] = 2.*np.pi - phidist[neg]
        if np.any(phidist > 2.*np.pi):
            gt = (phidist > 2.*np.pi)
            phidist[gt] -= 2.*np.pi
    phidist *= 180./np.pi

    plsamples = np.array(samples)
    plsamples[:,3] *= 180./np.pi
    extra = np.column_stack((plsamples,Mdist,ddist,Bdist,phidist))
    labels = ('$x_0$/arcsec','$y_0$/arcsec','$V$/arcsec/yr','$PA_V/^\circ$',
              '$M_\star/M_\odot$','$d/pc$','$B$','$\phi/^\circ$')
    fig = corner.corner(extra,labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True)
    fig.savefig('inparams_tri.png')
    plt.close(fig)

    N_z = 500
    N_vz = 500

    # Define tested region of z, vz space
    z_vz_data = get_z_vz_data(R,V,B,N_z,N_vz)

    # parallel
    nrun = 1000000
    j = np.random.randint(len(samples),size=nrun)
    dist = np.random.normal(loc=d,scale=derr,size=nrun)
    mass = np.random.normal(loc=M,scale=Merr,size=nrun)
    R = np.sqrt(samples[j,0]**2 + samples[j,1]**2) * dist
    pa0 = np.arctan2(samples[j,1],samples[j,0])-np.pi/2. # from E of N
    V = samples[j,2] * dist
    B = V**2 * R / (8*np.pi**2 * mass)
    zsgn = np.ones(nrun)
    phi = samples[j,3] - pa0
    phi += 2.*np.pi
    # ensure phi is defined correctly
    while np.max(phi) >= np.pi:
        phi[(phi >= np.pi)] -= 2.*np.pi
    if np.any(phi < 0.):
        neg = (phi < 0.)
        zsgn[neg] = -1.
        phi[neg] = 2.*np.pi - phi[neg]
        if np.any(phi > 2.*np.pi):
            gt = (phi > 2.*np.pi)
            phi[gt] -= 2.*np.pi
    # swap sign of z when phi would have been negative
    z = z_vz_data['z_list'][np.random.randint(N_z,size=nrun)] * zsgn
    vz = z_vz_data['vz_list'][np.random.randint(N_vz,size=nrun)] * zsgn
    pars = []
    pars = [np.append(pars,[z[i],vz[i],R[i],V[i],B[i],phi[i]]) for i in range(nrun)]
    pool = Pool(processes=16)
    el = pool.map(calc_elements_array,pars)
    el = np.array(el)

    ok = (el[:,1] > 0 ) & (el[:,3] < 1) & (el[:,2] < 50)
    fig = corner.corner(el[ok],color='k',top_ticks=True,bins=50,
                        labels=('$a/au$','$q/au$','$Q/au$',
                                '$e$','$I/^\circ$','$\Omega/^\circ$',
                                '$\omega/^\circ$',r'$\varpi/^\circ$','$f/^\circ$'),
                        range=[1.,(0.,np.max(el[:,1])),1.,(0,1),(0,90),(0,360),(0,360),(0,360),(0,360)])
    fig.savefig('elem_corner.png')
    plt.close(fig)

# data
info = getsys('HD 206893')
t,N,Nerr,E,Eerr,M,Merr,d,derr = info
#print(info)

elem(t,N,Nerr,E,Eerr,M,Merr,d,derr)


# unused single processor version

##     if 0:
##         rvbphi = []
##         el = []
##         for i in range(100000):
            
##             # select random R, V, B, phi from samples
##             j = np.random.randint(len(samples))
##             dist = np.random.normal(loc=d,scale=derr)
##             mass = np.random.normal(loc=M,scale=Merr)
##             R = np.sqrt(samples[j][0]**2 + samples[j][1]**2) * dist
##             pa0 = np.arctan2(samples[j][1],samples[j][0]) # from x, not E of N
##             V = samples[j][2] * dist
##             B = V**2 * R / (8*np.pi**2 * mass)
##             zsgn = 1.
##             phi = samples[j][3] - pa0
##         #    print(samples[j][3],pa0,phi)
##             # ensure phi is defined correctly
##             while phi >= np.pi: phi -= 2.*np.pi
##         #    print(phi)
##             if -np.pi < phi < 0.:
##                 zsgn = -1.
##                 phi = 2.*np.pi - phi
##                 if phi > 2.*np.pi: phi -= 2.*np.pi
##         #    print(j,R,V,B,phi,pa0,zsgn)
##             # swap sign of z when phi would have been negative
##             z = z_vz_data['z_list'][np.random.randint(N_z)] * zsgn
##             vz = z_vz_data['vz_list'][np.random.randint(N_vz)] * zsgn
            
##             # compute
##             a = calc_elements_array([z,vz,R,V,B,phi])
##             if len(el) == 0:
##                 el = np.append(el,a)
##                 rvbphi = np.append(rvbphi,[R,V,B,phi,zsgn,z,vz])
##             else:
##                 el = np.vstack((el,a))
##                 rvbphi = np.vstack((rvbphi,[R,V,B,phi,zsgn,z,vz]))
