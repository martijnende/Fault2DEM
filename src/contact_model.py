"""

"""

import numpy as np
from math import sqrt


class ContactModel:

    def __init__(self):
        pass

    def init_step(self):
        f = np.zeros((self.sim_params["N"], 2))
        self.particles["f"] = f

    def update_forces(self):
        self.init_step()
        self.update_wall_forces()
        self.update_body_forces()
        self.update_particle_forces()
        self.add_damping()


    def update_particle_forces(self):
        """
        Compute particle-particle interaction forces
        """

        N = self.sim_params["N"]
        coords = self.particles["coords"]
        f = self.particles["f"]
        stiff = self.particles["stiffness"]
        r = self.particles["radius"]

        inds = self.contacts["inds"]
        ptrs = self.contacts["ptrs"]
        lookup = self.contacts["lookup"]
        offsets = self.contacts["offsets"]
        N_all = len(lookup)

        for n in range(N_all):
            neighbours = ptrs[inds[n]:inds[n+1]]
            i = lookup[n]
            x1, y1 = coords[i] + offsets[n]
            r1 = r[i]

            for k in neighbours:
                j = lookup[k]
                x2, y2 = coords[j] + offsets[k]
                r2 = r[j]

                d_sq = (x1 - x2)**2 + (y1 - y2)**2
                delta_sq = (r1 + r2)**2 - d_sq
                if delta_sq > 0:
                    inv_d = 1.0 / sqrt(d_sq)
                    nx = (x1 - x2)*inv_d
                    ny = (y1 - y2)*inv_d

                    delta = sqrt(delta_sq)
                    fn = stiff * delta

                    f[i][0] += fn * nx
                    f[i][1] += fn * ny
        pass

    def update_body_forces(self):
        """
        Add body forces (gravity or viscous drag) to particles forces
        """

        f = self.particles["f"]
        v = self.particles["v"]
        m = self.particles["mass"]
        coords = self.particles["coords"]
        dx = self.particles["shear_dist"]
        y_max = coords[:, 1].max()

        # Gravitational pull
        if np.abs(self.sim_params["gravity"]) > 0:
            f[:, 1] += self.sim_params["gravity"] * m

        # Shear drag
        if self.sim_params["drag"] * self.sim_params["strain_rate"] > 0:
            # Compute position-based load-point velocity
            # (zero at middle of the sample)
            v_lp = self.sim_params["strain_rate"] * (coords[:, 1] - 0.5*y_max)

            # Update load-point displacements
            dx += (v_lp - v[:, 0]) * self.dt

            # Add shear drag
            f[:, 0] += self.sim_params["drag"] * dx

        pass

    def update_wall_forces(self):
        """Add particle-wall interaction forces"""

        coords = self.particles["coords"]
        f = self.particles["f"]
        k = self.particles["stiffness"]
        r = self.particles["radius"]
        z_max = self.sim_params["box"][1]

        # Find particles near the top and bottom walls
        bottom_particles = (coords[:, 1] < r)
        top_particles = (coords[:, 1] > z_max - r)

        # Add particle-wall interactions
        f[bottom_particles, 1] += k*(r[bottom_particles] - coords[bottom_particles, 1])
        f[top_particles, 1] += k*(z_max - r[top_particles] - coords[top_particles, 1])

        pass

    def add_damping(self):
        """
        Artificially alter particle forces to drain energy from the system
        (similar to restitution coefficient < 1)
        """
        f = self.particles["f"]
        v = self.particles["v"]
        sign_v = np.sign(v)

        f -= self.sim_params["damping_factor"]*np.abs(f)*sign_v

        pass

