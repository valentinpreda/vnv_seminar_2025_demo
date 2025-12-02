import jax
import jax.numpy as jnp
from jax import random
from jax.numpy.linalg import norm
from jax.numpy import array, concatenate, sqrt, stack, exp
import matplotlib.pyplot as plt
import time

# constants
dt = 10  # simulation step size
SIM_STEPS = round(24*3600/dt) # nr of simulation steps
mu = 3.986004418e14  # Gravitational parameter of Earth (mu = GM) [m^3/s^2]
Re = 6.3781363e6  # Earth's mean radius [m]
J2 = 1.08263e-3  # Earth's second zonal harmonic (dimensionless)
m = 10.  # Spacecraft mass [kg]
Cd = 2.3  # Drag coefficient (dimensionless)
A_nom = 0.7  # Spacecraft cross-sectional area [m^2]
rho0 = 1.225  # Atmospheric density at sea level [kg/m^3]
H = 8500.  # Atmospheric scale height [m]

# continuous dynamics
def f(x, u, p):
  r = x[0:3] # position in ECI (m)
  v = x[3:6] # velocity in ECI (m/s)

  # kepler acceleration
  r_norm = norm(r)
  a_kepler = -mu*r/r_norm**3

  # J2 acceleration
  z2 = r[2]**2
  factor = 1.5 * J2 * mu * Re**2 / r_norm**7
  ax = factor * r[0] * (5*z2 - r_norm**2)
  ay = factor * r[1] * (5*z2 - r_norm**2)
  az = factor * r[2] * (5*z2 - 3*r_norm**2)
  a_J2 = stack([ax, ay, az])

  # atmospheric drag acceleration
  h = norm(r) - Re
  rho = rho0 * exp(-h / H)
  A = A_nom + 0.6*p[1]
  a_drag = -0.5 * rho * Cd * A / m * norm(v) * v
  
  # acceleration due to random forces
  a_random = 5e-5 * u

  # total acceleration
  vdot = a_kepler + a_J2 + a_drag + a_random

  # state derivative
  xdot = concatenate([v, vdot])
  return xdot


def step(x, u, p):
  # RK4 step
  k1 = f(x, u, p)
  k2 = f(x + 0.5*dt*k1, u, p)
  k3 = f(x + 0.5*dt*k2, u, p)
  k4 = f(x + dt*k3, u, p)
  x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

  # output altitude position in km
  y = (norm(x[0:3]) - Re) / 1e3

  return x_next, y


def sim(p, seed):
  # parametrized initial conditions, nominally circular orbit at 240km altitude
  r0x = Re + 240e3
  v0y = sqrt(mu/r0x)
  r0 = array([r0x, 0, 0]) 
  v0 = array([0, v0y+1e-2*p[0] , 0]) 
  x0 = concatenate([r0, v0])

  # generate zero mean, unit variance random signals 
  rng = random.key(seed)
  us = random.normal(rng, (SIM_STEPS, 3))

  # effient way to perform simulation steps and collect outputs
  _, ys = jax.lax.scan(lambda x, u: step(x, u, p), x0, us)
  
  return ys

def monte_carlo(batch_size):
  # vectorize the simulation function
  sim_batch = jax.vmap(sim)

  # different seed for each simulation
  seed_batch = jnp.arange(batch_size) 

  # sample random parameter values 
  rng = random.key(batch_size)
  p_batch = 2*jax.random.uniform(rng, (batch_size, 2))-1 # uniformly distributed in [-1 1]

  # run batch of simulations 
  ys_batch = sim_batch(p_batch, seed_batch)
  
  return ys_batch


def plot(ys_batch):
  plt.figure(figsize=(8, 8))

  t = jnp.linspace(0, SIM_STEPS*dt, SIM_STEPS) / 3600 # time vector in hours

  # plot only 1000 trajectories for clarity
  plt.plot(t, ys_batch[:1000, :].T, color='b', alpha=.01)
  
  plt.xlabel('Time (h)')
  plt.ylabel('Altitude (km)')

  plt.show()
  plt.close()


if __name__ == "__main__":
  start_time = time.time()
  ys_batch = monte_carlo(10000)
  end_time = time.time()
  
  print(f"Monte Carlo simulation time: {end_time - start_time} seconds")
  plot(ys_batch)