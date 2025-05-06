import h5py
import numpy as np
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from helper import build_pdf, time_average, interp
import json

# Latex fonts
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsfonts}')

with open('parameters.json') as f:
    params = json.load(f)

def label_subplot(ax, string, xy=(0.0, 1.05)):
    ax.annotate(string, xy=xy, xycoords='axes fraction')
    
fname = './timeseries/timeseries_s1.h5'

with h5py.File(fname, mode='r') as file:
    t = file['tasks']['<wb>'].dims[0]['sim_time'][:]
    WB = file['tasks']['<wb>'][:].squeeze()

# Check volume-averaged buoyancy flux
plt.plot(t,WB)
plt.show()    

# Define transient period to discard
t0 = 100

# Create empty set of sample distributions for internal and boundary points
f_Xs = [None]*3
df_Xs = [None]*3

fname = './timeseries/timeseries_s1.h5'

# Get 1d profiles to define f_X
with h5py.File(fname, mode='r') as file:
    t = file['tasks']['<wb>(z)'].dims[0]['sim_time'][:]
    z_1d = file['tasks']['<wb>(z)'].dims[2][0][:]

    # Unnormalised sample distributions based on buoyancy flux
    f_Xs[0] = -params['alpha_2']*time_average( file['tasks']['<db_dz>(z)'][:], t, t0)
    f_Xs[1] =  time_average( file['tasks']['<wb>(z)'][:], t, t0)
    df_Xs[0] = -params['alpha_2']*time_average( file['tasks']['<ddb_dz2>(z)'][:], t, t0)
    df_Xs[1] = time_average(file['tasks']['<dwb_dz>(z)'][:], t, t0)

# Define grey
col = (0.85,0.85,0.85)

# Top half of the domain
top = z_1d>params['Lz']/2

# Split the diffusive buoyancy flux between the bottom and top boundary layer
f_Xs[2] = np.zeros(z_1d.shape)
df_Xs[2] = np.zeros(z_1d.shape)
f_Xs[2][top] = f_Xs[0][top]
df_Xs[2][top] = df_Xs[0][top]
f_Xs[0][top] = 0
df_Xs[0][top] = 0

# Get field snapshots for f_Y
fname = './fields/fields_s1.h5'

# Number of uniformly spaced cells to use for interpolation
Nz = 256
dz = 1/Nz
z = np.linspace(dz/2,1-dz/2,Nz)

# Get boundary information from the wall slices  
with h5py.File(fname, mode='r') as file:
    t = file['tasks']['b(z=0)'].dims[0]['sim_time'][:]
    Z = file['tasks']['b'].dims[2][0][:] # Chebyshev
    X = file['tasks']['b'].dims[1][0][:]
    B = file['tasks']['b'][:]
    U = file['tasks']['u'][:]
    grad_B = file['tasks']['grad_b'][:]
    grad_U = file['tasks']['grad_u'][:]

dx = (params['Lx']/B.shape[1], params['Lz']/B.shape[2])

# Defined normalisation constant
N = np.trapz(f_Xs[0]+f_Xs[1]+f_Xs[2], x=z_1d)

# Pick an index for the snapshot (after negative fluctuation in buoyancy at the bottom boundary)
idx = np.where(t>t0)[0][2]
fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios':[6,1],'wspace':0.05,'bottom':0.2},figsize=(9,2))

# Plot field
ax[0].contourf(X, Z, B[idx,:,:].T, np.linspace(-0.5, 0.5, 21), cmap='RdBu_r', extend='both')
ax[0].set_aspect('equal', adjustable='box')
ax[0].contourf(X, Z, 0*B[idx,:,:].T, [0,1e5], cmap='gray', alpha=0.3)
ax[0].contour(X, Z, U[idx,1,:,:].T,np.linspace(-1,1,21), colors='w',linewidths=0.6)
ax[0].set_xlabel('$X^{1}$')
ax[0].set_ylabel('$X^{2}$')
ax[0].set_xlim([0,params['Lx']])
ax[0].set_ylim([0,params['Lz']])
label_subplot(ax[0],'$(a)$')

# Plot boundary layer and bulk zones
ax[1].fill_betweenx(z_1d, f_Xs[0]/N, color=col, alpha=0.5, edgecolor=None)
ax[1].fill_betweenx(z_1d, f_Xs[1]/N, color=col, alpha=0.5, edgecolor=None)
ax[1].fill_betweenx(z_1d, f_Xs[2]/N, color=col, alpha=0.5, edgecolor=None)
ax[1].plot((f_Xs[0]+f_Xs[1]+f_Xs[2])/N, z_1d, '-C0', lw=0.6)
ax[1].set_xlabel('$f_{X|Z}\mathbb{P}(Z)$')
ax[1].set_yticks([])
ax[1].set_ylim([0,1])
ax[1].set_xlim([0,1.1])
label_subplot(ax[1],'$(b)$')
plt.tight_layout()

plt.savefig('fig_field.pdf',dpi=300)
plt.show()

# Define fields for constructing the PDF
Y1 = interp(U[t>t0,1,:,:], Z, z, axis=2)
Y2 = interp(B[t>t0,:,:], Z, z, axis=2)

# Note the dimensions of grad_U are [:, (d/dx,d/dy), (u, v), :, :]
grad_Y1 = interp(grad_U[t>t0,1,1,:,:], Z, z, axis=2)
grad_Y2 = interp(grad_B[t>t0,1,:,:], Z, z, axis=2)

# Initialise a list of PDF objects
Omega = [None]*3
dOmega = [None]*3

minmax = [(Y1.min(), Y1.max()), (Y2.min(), Y2.max())]
yrange = [minmax, [(-1,1),(-0.5,1.5)], minmax]

for i in [0,1,2]:
    print('i = ',i)

    # Interpolate the (unnormalised) sample distributions to a uniform grid
    f_X =  np.tile( interp(f_Xs[i],  z_1d, z, axis=0), (Y1.shape[1],1) )
    df_X = np.tile( interp(df_Xs[i], z_1d, z, axis=0), (Y1.shape[1],1) )

    # Construct PDF object
    Omega[i] = build_pdf(Y1, Y2, f_X=f_X, df_X=df_X, dx=dx, range_=yrange[i] ) 

# Limits for the plots
y1lim = [-0.16,0.16]
y2lim = [[-1.5,2.5],[-0.5,0.5],[-0.7,0.7]]
xlim_top = [[-1/3,1],[-1,3],[-1,1]]
xlim_bot = [[-1/3,1],[-1,3],[-3.2,3.2]]

fig, ax = plt.subplots(1,3,gridspec_kw={'width_ratios':[1,1.2,1],'wspace':0.3,'bottom':0.2},figsize=(7,3.5))
labs = ['$(a)$','$(b)$','$(c)$']

# First plot the marginal distributions and budget terms for the boundary layers
for i in [0,2]:

    # Calculate coefficients in the forward equation
    D_1 =  Omega[i].E_Y2( Omega[i].E_Y(-params['alpha_2']*grad_Y2, boundary=True) )
    V   = -Omega[i].E_Y2( Omega[i].E_Y(Y1, boundary=True) )

    ax[i].barh(Omega[i].y2, Omega[i].f_Y2(),color='darkred',alpha=0.8,height=Omega[i].dy[1])
    ax[i].set_xlabel('$f_{Y^{2}}$')
    ax[i].set_ylim(y2lim[i])
    ax[i].set_xlim(xlim_bot[i])
    if i==0: ax[i].set_ylabel('$y^{2}$')
    
    ax2 = ax[i].twiny()
    ax2.fill_betweenx(Omega[i].y2,-V * Omega[i].f_Y2(), color=col, alpha=0.6,lw=0)
    ax2.plot(-V * Omega[i].f_Y2(), Omega[i].y2, '-k', lw=0.6)
    ax2.plot(-np.gradient(D_1 * Omega[i].f_Y2(),Omega[i].y2), Omega[i].y2,'-',color='navy',lw=0.8)
    ax2.set_xlabel('$\partial_{t} f_{Y^{2}}$')
    ax2.set_xlim(xlim_top[i])
    ax2.plot([0,0],y2lim[i],'-k',lw=0.6)
    ax2.plot(xlim_top[i],[y2lim[1][0]]*2,'-k',lw=0.6)
    ax2.plot(xlim_top[i],[y2lim[1][1]]*2,'-k',lw=0.6)
    label_subplot(ax[i], labs[i], xy=(0.0, 1.1))
    
i = 1 # bulk zone

# Calculate coefficients in the forward equation
D_1 = np.array([Omega[i].E_Y(-params['alpha_1']*grad_Y1, boundary=True),
                Omega[i].E_Y(-params['alpha_2']*grad_Y2, boundary=True)])
V = -Omega[i].E_Y(Y1, boundary=True)

# Then plot the joint distribution and budget terms for the bulk

c = 0.1 # Constant for positive and negative forcing zones
ax[i].contourf(Omega[i].y1_2d, Omega[i].y2_2d, Omega[i].f_Y,cmap='pink_r')
ax[i].contourf(Omega[i].y1_2d, Omega[i].y2_2d, -V*Omega[i].f_Y, [c,np.inf], colors='k',alpha=0.2)
ax[i].contourf(Omega[i].y1_2d, Omega[i].y2_2d, -V*Omega[i].f_Y, [-np.inf, -c], colors='w',alpha=0.5)
ax[i].contour(Omega[i].y1_2d, Omega[i].y2_2d, -V*Omega[i].f_Y, [c], colors='k', linestyles='solid', linewidths=0.4)
ax[i].contour(Omega[i].y1_2d, Omega[i].y2_2d, -V*Omega[i].f_Y, [-c], colors='k', linestyles='solid', linewidths=0.4)
ax[i].quiver(Omega[i].y1_2d, Omega[i].y2_2d, D_1[0]*Omega[i].f_Y, D_1[1]*Omega[i].f_Y, angles='xy', scale_units='xy', scale=0.7, width=2e-3,color='navy',headwidth=4)

ax[i].annotate(r'$\underline{\uparrow}$', (0.05,0.4))
ax[i].annotate(r'$\underline{\downarrow}$', (-0.08,0.2))
ax[i].annotate(r'$\overline{\uparrow}$', (0.1,0.14))
ax[i].annotate(r'$\overline{\downarrow}$', (-0.09,-0.4))

ax[i].set_xlim(y1lim)
ax[i].set_xlabel('$y^{1}$')
ax[i].set_ylim(y2lim[i])
label_subplot(ax[i], labs[i], xy=(0.0, 1.1))
plt.tight_layout()

plt.savefig('fig_joint.pdf',dpi=300)

plt.show()

