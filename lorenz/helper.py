import numpy as np
import matplotlib.pyplot as plt

""" Helper routines to investigate the system
Y1 = c0.a0(t).cos(k.X1).sin(pi.X2)
Y2 = c1.a1(t).cos(k.X1).sin(pi.X2) - c2.a2(t).sin(pi.X2).cos(pi.X2)
"""

debug_level = 'ignore'

def label_subplot(s):
    """Plotting tool"""
    plt.annotate(s, xy=(0.0, 1.05), xycoords='axes fraction')

def tangent(t, a, s=10, r=28, b=8/3):
    """Phase space velocity of the Lorenz equations."""
    da = [0]*3
    da[0] = s * (a[1]-a[0])
    da[1] = a[0] * (r - a[2]) - a[1]
    da[2] = a[0] * a[1] - b * a[2]
    return da

def calc_coeffs(r=28,b=8/3):
    k = np.sqrt( (4-b)*np.pi**2/b )
    c0 = np.sqrt(2)/np.pi*(k**2+np.pi**2)
    c1 = np.sqrt(2)/np.pi/r
    c2 = 2/r/np.pi
    return [c0, c1, c2]

def wavenumber(b=8/3):
    """wavenumber k for a given parameter b"""
    return np.sqrt( (4-b)*np.pi**2/b )

def field(X, a, s=10,r=28,b=8/3):
    """Return the vertical velocity and buoyancy field for a given state a(t)"""
    k = wavenumber(b)
    c = calc_coeffs(r=r,b=b)

    # Vertical velocity
    Y0 = c[0]*a[0]*np.cos(k*X[0])*np.sin(np.pi*X[1])

    # Buoyancy field
    Y1 = c[1]*a[1]*np.cos(k*X[0])*np.sin(np.pi*X[1])-c[2]*a[2]*np.sin(2*np.pi*X[1])/2
    return [Y0, Y1]

def jacobian(X, a, s=10,r=28,b=8/3):
    """Return the jacobian field for a given state a(t) in terms of X"""
    k = wavenumber(b)
    c = calc_coeffs(r=r,b=b)

    dY0d0 = -k*c[0]*a[0]   * np.sin(k*X[0]) * np.sin(np.pi*X[1])
    dY0d1 = np.pi*c[0]*a[0]* np.cos(k*X[0]) * np.cos(np.pi*X[1])

    dY1d0 = -k*c[1]*a[1]*np.sin(k*X[0])*np.sin(np.pi*X[1])
    dY1d1 = np.pi*c[1]*a[1]*np.cos(k*X[0])*np.cos(np.pi*X[1])-np.pi*c[2]*a[2]*np.cos(2*np.pi*X[1])

    J = dY0d0*dY1d1 - dY0d1*dY1d0
    return J

def jac(y, a, r=28,b=8/3, domain='D1'):
    """Return the jacobian field for a given state a(t) in terms of y"""
    # Note that J(y) is needed in order to construct the probability density as
    # a function of y
    
    k = wavenumber(b)
    c = calc_coeffs(r=r,b=b)

    # Let S=sin(pi.X2)
    # Then S**2.(1-S**2) = R, where
    R = 1/(c[2]*a[2])**2 * (c[1]*a[1]/(c[0]*a[0])*y[0] - y[1])**2
    
    dRd1  =  c[1]*a[1]/(c[0]*a[0])/(c[2]*a[2])**2 * (c[1]*a[1]/(c[0]*a[0])*y[0] - y[1])
    dRd2  = -1/(c[2]*a[2])**2 * (c[1]*a[1]/(c[0]*a[0])*y[0] - y[1])

    # Distinguish two subdomains over which Y is invertible
    # D1:=(0,pi/2]x[0,pi/4)     (0<S<=1/sqrt(2))
    # D2:=(0,pi/2]x[pi/4,pi/2]  (1/sqrt(2)<S<=1)
    
    if (domain == 'D1'):
        sign = -1
    else:
        sign = 1

    # S=sin(pi.X2)
    # C=cos(k.X1)

    # The inversion
    # nan = np.sqrt(-1)=0**(-1) warnings suppressed
    with np.errstate(invalid=debug_level):
        S   = np.sqrt( (1 + sign*np.sqrt(1-4*R))/2)

        C   = y[0] / (c[0]*a[0]) / S

        # Derivatives to compute jacobian
        dSdR = -sign*1/2 * S**(-1) * (1-4*R)**(-1/2)
        dCdR = -(y[0] / (c[0]*a[0]) /S**2) * dSdR

        dZdS = 1/np.pi * 1 / np.sqrt(1 - S**2)
        dXdC = -1/k * 1 / np.sqrt(1 - C**2)

        dXd1 = dXdC * (1/(c[0]*a[0])/S + dCdR * dRd1)
        dXd2 = dXdC * dCdR * dRd2
        dZd1 = dZdS * dSdR * dRd1
        dZd2 = dZdS * dSdR * dRd2

    J = np.abs(dXd1*dZd2 - dZd1*dXd2)
    J[np.isnan(J)] = 0
    return J*k / np.pi


def calc_jac(y,A, r=28, b=8/3):
    """Calculate jacobian in terms of Y by combining jacobians for D1 and D2"""
    return 2*(jac(y,A,r=r, b=b, domain='D1') + jac(y,A,r=r,b=b, domain='D2'))

def Kmax(s=10, r=28, b=8/3):
    """Useful bounds for plotting."""
    if b <= 2:
        return (r+s)**2
    else:
        return b**2*(r+s)**2/(4*(b-1))

def Jmax(s=10, r=28, b=8/3):
    """Useful bounds for plotting."""
    if b <= 2:
        return r**2
    else:
        return (b*r)**2/(4*(b-1))

def Zlim(s=10, r=28, b=8/3):
    """Useful bounds for plotting."""
    J = Jmax(s,r,b)
    return [r-np.sqrt(J), r+np.sqrt(J)]
    
def Xlim(s=10, r=28, b=8/3):
    """Useful bounds for plotting."""
    Zmax = max( Zlim(s,r,b) )
    return [-np.sqrt(2*s*Zmax), np.sqrt(2*s*Zmax)]

def Ylim(s=10, r=28, b=8/3):
    """Useful bounds for plotting."""
    J = Jmax(s,r,b)
    return [-np.sqrt(J), np.sqrt(J)]

def E(y, a, G):
    """Expectation of G. Values of G for branches D1 and D2 given in list G = [G[0],G[1]]."""
    return jac(y,a, domain='D1')*G[0] + jac(y,a, domain='D2')*G[1]

def rhs(y, c, a):
    # Let S=sin(pi.X2)
    # Then S**2.(1-S**2) = R, where
    return 1/(c[2]*a[2])**2 * (c[1]*a[1]/(c[0]*a[0])*y[0] - y[1])**2

def inversion(y, c, a, R, domain='D1'):
    # Return S=sin(pi.X2) when R=rhs(...) is given
    with np.errstate(invalid=debug_level):
        if domain == 'D1':
            S = np.sqrt( (1 - np.sqrt(1-4*R))/2)
        else:
            S = np.sqrt( (1 + np.sqrt(1-4*R))/2)
    C = y[0] / (c[0]*a[0]) / S
    return [C,S]
    
def calc_dydt(y, a, r=28, b=8/3):
    k = np.sqrt( (4-b)*np.pi**2/b )
    c = calc_coeffs(r)

    # Get tangent vector
    dYdt = tangent(0,a)

    R = rhs(y,c,a) # 1/(c3*A[2])**2 * (c2*A[1]/(c1*A[0])*Y[0] - Y[1])**2

    C,S = inversion(y,c,a,R, domain='D1')
    # Zh   = np.sqrt( (1 + sign*np.sqrt(1-4*P))/2)
    # Xh   = Y[0] / (c[0]*A[0]) / Zh

    d1dt = [y[0]*dYdt[0]/a[0]]
    d2dt = [0*d1dt[0]]

    thresh = np.sqrt(2)/np.pi/r*a[1]*C*S
    js = y[1]>=thresh
    d2dt[0][js] = thresh[js]*dYdt[1]/a[1] + 1/r/np.pi*dYdt[2]*2*S[js]*np.sqrt(1-S[js]**2)
    js = y[1]<thresh
    d2dt[0][js] = thresh[js]*dYdt[1]/a[1] - 1/r/np.pi*dYdt[2]*2*S[js]*np.sqrt(1-S[js]**2)
    
    C,S = inversion(y,c,a,R, domain='D2')
    # Zh   = np.sqrt( (1 + sign*np.sqrt(1-4*P))/2)
    # Xh   = y[0] / (c[0]*A[0]) / Zh
    d1dt += [y[0]*dYdt[0]/a[0]]
    d2dt += [0*d1dt[0]]
    js = y[1]>=thresh
    d2dt[1][js] = thresh[js]*dYdt[1]/a[1] + 1/r/np.pi*dYdt[2]*2*S[js]*np.sqrt(1-S[js]**2)
    js = y[1]<thresh
    d2dt[1][js] = thresh[js]*dYdt[1]/a[1] - 1/r/np.pi*dYdt[2]*2*S[js]*np.sqrt(1-S[js]**2)

    return E(y, a, d1dt), E(y, a, d2dt)


def plot_algebraic_curve(a, n=500, r=28, ocol='k', icol='k', **kwargs):
    """Plot singularities of probability distribution in phase space."""
    c = calc_coeffs(r)

    s = np.linspace(-1,1,n)
    Ws0 = c[0]/np.sqrt(2)*a[0]*s

    s = np.linspace(0,1,n)
    Ws1 = c[0]*a[0]*s

    plt.plot(Ws0, c[1]/c[0]*a[1]/a[0]*Ws0 + c[2]/2*a[2],color=ocol,**kwargs)
    plt.plot(Ws0, c[1]/c[0]*a[1]/a[0]*Ws0 - c[2]/2*a[2],color=ocol,**kwargs)
    plt.plot(Ws1, c[1]/c[0]*a[1]/a[0]*Ws1 - c[2]*a[2]*s*(1-s**2)**(1/2),color=icol,**kwargs)
    plt.plot(Ws1, c[1]/c[0]*a[1]/a[0]*Ws1 + c[2]*a[2]*s*(1-s**2)**(1/2),color=icol,**kwargs)
    plt.plot(-Ws1, -c[1]/c[0]*a[1]/a[0]*Ws1 - c[2]*a[2]*s*(1-s**2)**(1/2),color=icol,**kwargs)
    plt.plot(-Ws1, -c[1]/c[0]*a[1]/a[0]*Ws1 + c[2]*a[2]*s*(1-s**2)**(1/2),color=icol,**kwargs)

