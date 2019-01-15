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
t = np.linspace(0, 1, 100)
dt = t[1] - t[0]

# Particle properties
r = 1.0

coords = np.array([
    [1.5, 1.0],
    [1.5, 2.9],
    [1.0, 5.0],
    [2.0, 8.0],
])
N_particles = coords.shape[0]
offsets = np.zeros(N_particles)
radius = np.ones(N_particles) * r
v = np.zeros((N_particles, 2))

Delta = coords[1, 1] - coords[0, 1]
rc_sq = r**2 - 0.25*Delta**2

# Simulation parameters
sim_params = {
    "N": N_particles,
    "box": np.array([3, 9]),
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
    "Z_ps": 1e-8,
}

output_step = 1
DEM.dt = dt
DEM.sim_params = sim_params
DEM.particles = particles
DEM.saved_data = []

DEM.render_mesh()
# DEM.draw_scene(tri=True)

for i in range(len(t)):
    DEM.t = t[i]
    DEM.init_step()
    DEM.update_particle_forces()

    if i % output_step == 0:
        data = {
            "particles": copy.deepcopy(DEM.particles),
            "contacts": copy.deepcopy(DEM.contacts),
        }
        DEM.saved_data.append(data)


f = -np.array([data["particles"]["f"][0, 1] for data in DEM.saved_data])
f0 = particles["stiffness"] * np.sqrt(4*r**2 - Delta**2)
f_ana = f0 * np.exp(-2*particles["stiffness"] * particles["Z_ps"]*t / (np.pi * rc_sq**2))

plt.plot(t, f)
plt.plot(t, f_ana, "k--")
plt.show()

