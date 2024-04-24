
"""Create plot of (a) vertical velocity and buoyancy modes for convection and
(b) the corresponding joint probability density, both evolving according to the
Lorenz (1963) model."""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mpl
from matplotlib import rc
import helper as lorenz

rc('text', usetex=True)

# Note that the default parameters for all functions are
# s = 10
# r = 28
# b = 8/3

# Calculate coefficients of the lorenz fields Y1 (vertical veloicty) and Y2 (buoyancy)
c = lorenz.calc_coeffs()

# Wavenumber for given b
k = lorenz.wavenumber()

# Simulate lorenz system
nt = 5000
tend = 200
dt = tend/nt
t = np.linspace(0,tend,nt+1)
sol = solve_ivp(lorenz.tangent, np.array([0,tend]), np.array([1.0,1.0,8.0]), t_eval=t)
a = sol.y.T

# Pick a time index to display fields
i = 2800

# Set bounds for the domain of PDF f
y1_min = -np.sign(a[i,0])*c[0]*a[i,0]
y1_max = np.sign(a[i,0])*c[0]*a[i,0]
y2_min = c[1]/np.sqrt(2)*a[i,1]/a[i,0] - c[2]/2*a[i,2]
y2_max = c[1]/np.sqrt(2)*a[i,1]/a[i,0] + c[2]/2*a[i,2]

n1 = 1000
n2 = 1000

scale = 1.2
y1_ = np.linspace(scale*y1_min, scale*y1_max, n1)
y2_ = np.linspace(scale*y2_min, scale*y2_max, n1)

d1 = (y1_max - y1_min)/n1
d2 = (y2_max - y2_min)/n2

# Y-domain for PDF
[y1,y2] = np.meshgrid(y1_,y2_, indexing='ij')

# Calculate the jacobian in terms of y1 and y2
J = lorenz.calc_jac([y1, y2], a[i,:])
Ed1dt, Ed2dt = lorenz.calc_dydt([y1,y2],a[i,:])

# Ignore nan = 0**(-1)
with np.errstate(invalid=lorenz.debug_level):
    Ed1dt = Ed1dt/J
    Ed2dt = Ed2dt/J

# Define X-domain and Y1, Y2 fields
x = np.linspace(0, 2*np.pi/k,200)
z = np.linspace(0, 1,200)
[X,Z] = np.meshgrid(x,z, indexing='ij')
Y1,Y2 = lorenz.field([X,Z], a[i,:])
dYdX = lorenz.jacobian([X,Z], a[i,:])

# Plot the results
cmap = mpl.colormaps['pink']
color = 'black'

plt.figure(figsize=(6,5))
plt.subplot(2,1,2)
plt.pcolor(y1, y2, J,cmap='pink_r')
plt.clim([0,0.06])
plt.streamplot(y1.T, y2.T, Ed1dt.T, Ed2dt.T, density=2, color='grey', linewidth=0.4, arrowstyle='->')
lorenz.plot_algebraic_curve(a[i,:], lw=0.8, icol=color,ocol=color)
lorenz.plot_algebraic_curve(a[i+1,:], icol=color,ocol=color, linestyle='--', lw=0.8, dashes=(8,3) )
plt.xlabel('$y_{1}$')
plt.ylabel('$y_{2}$')
lorenz.label_subplot('$(b)$')

plt.subplot(2,1,1)
plt.contourf(X,Z,Y2, cmap='RdBu_r')
plt.gca().set_aspect('equal', adjustable='box')
plt.contour(X,Z,Y1,10, colors='w',linewidths=1.0)
plt.contourf(X,Z,dYdX,[-1000,0,1000], cmap='gray', alpha=0.4)
plt.plot([np.pi/k,np.pi/k],[0,1],color=color,lw=1.0)
plt.plot([0,2*np.pi/k],[0.25,0.25],color=color,lw=1.0)
plt.plot([0,2*np.pi/k],[0.75,0.75],color=color,lw=1.0)
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
lorenz.label_subplot('$(a)$')
plt.tight_layout()

# plt.savefig('fig_lorenz.png', dpi=300)

plt.show()

