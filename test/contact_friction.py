"""
Test suite for the contact friction model

Tests:
[OK]    mu_ss = mu_ref + a*log(v/v_ref)
[OK]    stress relaxation when vs = 0
[OK]    remeshing ok?
[OK]    reverse motion
[OK]    independent of normal force
[OK]    shear rotation onto contact plane
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sinh, asinh
from src.artist import Rembrandt
from src.contact_model import ContactModel
from src.contact_detection import MeshDetection
from src.helper import Elves
import copy


class TestSuite(ContactModel, Elves, MeshDetection, Rembrandt):

    def __init__(self):
        ContactModel.__init__(self)
        Elves.__init__(self)
        MeshDetection.__init__(self)
        Rembrandt.__init__(self)
        pass


DEM = TestSuite()

# Time vector
t = np.linspace(0, 2*np.pi, 10000)
dt = t[1] - t[0]

# Particle trajectory
R_p = 1.8
x_p = R_p*np.cos(np.pi - t) + 5
y_p = R_p*np.sin(np.pi - t) + 5
vx_p = R_p*np.sin(np.pi - t)
vy_p = -R_p*np.cos(np.pi - t)

# Particle properties
r = 1.0

coords = np.array([
    [0.0, 0.0],
    [-1.9, 0.0],
    [-3.0, 3.0],
    [3.0, 3.0]
]) + 5
N_particles = coords.shape[0]
offsets = np.zeros(N_particles)
radius = np.ones(N_particles) * r
v = np.zeros((N_particles, 2))

# Simulation parameters
sim_params = {
    "N": N_particles,
    "box": np.array([10, 10]),
}

# Particle parameters
particles = {
    "ID": np.arange(N_particles),
    "coords": coords,
    "offset": offsets,
    "radius": radius,
    "v": v,
    "r_max": 1,
    "r_min": 1,
    "stiffness": 1e7,
    "vc_ref": 1.0,
    "mu_ref": 0.6,
    "a_tilde": 0.005,
    "mu": 0.6,
    "Z_ps": 0.0,
}

output_step = 20
DEM.dt = dt
DEM.sim_params = sim_params
DEM.particles = particles
inv_sinh = 1.0 / sinh(particles["mu_ref"] / particles["a_tilde"])
DEM.particles["inv_sinh_mu_ref"] = inv_sinh
DEM.saved_data = []

DEM.render_mesh()
# DEM.draw_scene(tri=True)

s = np.zeros_like(t)
mu = np.zeros_like(t)
mu_ss = particles["a_tilde"]*asinh(vy_p[0] / (particles["vc_ref"] * inv_sinh))

for i in range(len(t)):
    DEM.t = t[i]
    DEM.particles["coords"][1] = x_p[i], y_p[i]
    DEM.particles["v"][1] = vx_p[i], vy_p[i]
    DEM.check_remesh()
    DEM.init_step()
    DEM.update_particle_forces()

    if i % output_step == 0:
        data = {
            "particles": copy.deepcopy(DEM.particles),
            "contacts": copy.deepcopy(DEM.contacts),
        }
        DEM.saved_data.append(data)

    sx, sy, delta = DEM.contacts["shear"][0]
    s[i] = np.sqrt(sx**2 + sy**2)
    fs, fn = DEM.contacts["forces"][0]
    mu[i] = fs / fn


DEM.dt *= 2
t2 = np.linspace(0, np.pi, 20000)
dt = t2[1] - t2[0]
DEM.dt = dt

# Particle trajectory
x_p = R_p*np.cos(np.pi - 10*t2) + 5
y_p = R_p*np.sin(np.pi - 10*t2) + 5
vx_p = 10*R_p*np.sin(np.pi - 10*t2)
vy_p = -10*R_p*np.cos(np.pi - 10*t2)

s2 = np.zeros_like(t2)
mu2 = np.zeros_like(t2)
mu_ss2 = particles["a_tilde"]*asinh(vy_p[0] / (particles["vc_ref"] * inv_sinh))
mu_ss2 = particles["mu_ref"] + particles["a_tilde"] * np.log(vy_p[0] / particles["vc_ref"])

for i in range(len(t)):
    DEM.t = t[i]
    DEM.particles["coords"][1] = x_p[i], y_p[i]
    DEM.particles["v"][1] = vx_p[i], vy_p[i]
    DEM.check_remesh()
    DEM.init_step()
    DEM.update_particle_forces()

    if i % output_step == 0:
        data = {
            "particles": copy.deepcopy(DEM.particles),
            "contacts": copy.deepcopy(DEM.contacts),
        }
        DEM.saved_data.append(data)

    sx, sy, delta = DEM.contacts["shear"][0]
    s2[i] = np.sqrt(sx**2 + sy**2)
    fs, fn = DEM.contacts["forces"][0]
    mu2[i] = fs / fn

# DEM.animate(tri=True)

plt.subplot(211)
plt.axhline(mu_ss, ls="--", c="k")
plt.axhline(mu_ss2, ls="--", c="k")
plt.plot(t, mu, ".-")
plt.plot(t2+t[-1], mu2, ".-")
plt.subplot(212)
plt.plot(t, s)
plt.plot(t2+t[-1], s2)
plt.show()

