# Prevent multi-threading upon initialising mpi4py
import os,mpi4py
os.environ["OMP_NUM_THREADS"] = "1"
mpi4py.rc.thread_level = 'single'

import numpy as np
import dedalus.public as d3
import logging
import json
logger = logging.getLogger(__name__)

# Parameters 
Ra = 1e7; Pr = 1
Lx, Lz = 4, 1
Nx, Nz = 512, 128

var = 1
lam = 1/2
mu_0 = 1/2
mu_1 = -1/2
sigma_0 = np.sqrt(2*lam*var)
sigma_1 = 0

dealias = 3/2
stop_sim_time = 4000
timestepper = d3.RK222
max_timestep = 2e-3 # 2e-3
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0,Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0,Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1',bases=xbasis)
tau_b2 = dist.Field(name='tau_b2',bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1',bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2',bases=xbasis)

# Substitutions
alpha_2 = (Ra * Pr)**(-1/2)
alpha_1 = (Ra / Pr)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

# Boundary conditions
g_1 = dist.Field(name='g_1')
g_0 = dist.Field(name='g_0')
g_1['g'] = mu_1
g_0['g'] = mu_0

kx_max = np.max(xbasis.wavenumbers)

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation('trace(grad_u) + tau_p = 0')
problem.add_equation('dt(u) - alpha_1*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)')
problem.add_equation('u(z=0) = 0')
problem.add_equation('u(z=1) = 0') # no-slip
problem.add_equation('integ(p) = 0') # Pressure gauge
problem.add_equation('dt(b) - alpha_2*div(grad_b) + lift(tau_b2) = -u@grad(b)')
problem.add_equation('b(z=1) = g_1')
problem.add_equation('b(z=0) = g_0') 

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial condition
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (1 - z) # Damp noise at walls
b['g'] += 1/2 - z # Add linear background


# Time Series
timeseries = solver.evaluator.add_file_handler('timeseries',iter=100)
timeseries.add_task(d3.Integrate((u@ez)*b)/(Lx*Lz),  layout='g', name='<wb>')

timeseries.add_task(d3.Integrate((u@ez)*b, ('x',))/Lx,  layout='g', name='<wb>(z)', scales=2)
timeseries.add_task(d3.Integrate(d3.grad((u@ez)*b)@ez, ('x',))/Lx,  layout='g', name='<dwb_dz>(z)', scales=2)
timeseries.add_task(d3.Integrate(grad_b@ez, ('x',))/Lx,  layout='g', name='<db_dz>(z)', scales=2)
timeseries.add_task(d3.Integrate(d3.grad(grad_b@ez)@ez, ('x',))/Lx,  layout='g', name='<ddb_dz2>(z)', scales=2)

# Boundary profiles
slices = solver.evaluator.add_file_handler('slices', sim_dt=0.2)

slices.add_task( grad_b(z=0)@ez,  layout='g', name='db_dz(z=0)')
slices.add_task( b(z=0),  layout='g', name='b(z=0)')
slices.add_task( grad_b(z=1)@ez,  layout='g', name='db_dz(z=1)')
slices.add_task( b(z=1),  layout='g', name='b(z=1)')


# Snapshots
fields = solver.evaluator.add_file_handler('fields', sim_dt=2)

fields.add_task( grad_b(z=0)@ez,  layout='g', name='db_dz(z=0)')
fields.add_task( b(z=0),  layout='g', name='b(z=0)')
fields.add_task( grad_b(z=1)@ez,  layout='g', name='db_dz(z=1)')
fields.add_task( b(z=1),  layout='g', name='b(z=1)')

fields.add_task(b,    name='b',scales=dealias)
fields.add_task(u,    name='u',scales=dealias)
fields.add_task(grad_b       , name='grad_b',scales=dealias)
fields.add_task(d3.grad(u),    name='grad_u',scales=dealias)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)

flow.add_property(d3.Integrate((u@ez)*b), name='<wb>')
flow.add_property(d3.Integrate(d3.grad(u@ez)@d3.grad(u@ez) + d3.grad(u@ex)@d3.grad(u@ex)), name='<|grad(u)|**2>')

np.random.seed(42)

params = {'Ra':Ra,'Pr':Pr,'Lx':Lx,'Lz':Lz,'alpha_1':alpha_1,'alpha_2':alpha_2,'mu_0':mu_0,'mu_1':mu_1,'sigma_0':sigma_0,'sigma_1':sigma_1,'lam':lam}

# Save parameters
with open('parameters.json', 'w') as f:
    json.dump(params, f, indent=2)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        dt = CFL.compute_timestep()

        # Calculate the Bcs for Y_t+1
        # Specify the bcs according to OU process

        B_0    = b(z=0).evaluate()['g'][0]
        g_0['g'] += lam * (mu_0 - B_0) * dt + sigma_0 * np.sqrt(dt) * np.random.normal()

        B_1    = b(z=1).evaluate()['g'][0]
        g_1['g'] += lam * (mu_1 - B_1) * dt + sigma_1 * np.sqrt(dt) * np.random.normal()
        
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            wb_avg = flow.grid_average('<wb>')
            du2 = flow.grid_average('<|grad(u)|**2>')
            
            logger.info('n={:4n}, t={:.4f}, dt={:.2e}, <wb>={:.3e}, alpha_1*<|grad(u)|**2>={:.3e}'.format(
                solver.iteration, solver.sim_time, dt, wb_avg, alpha_1*du2))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
