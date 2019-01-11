"""

"""

import numpy as np
from math import sqrt, sinh


class ContactModel:

    def __init__(self):
        pass

    def init_step(self):
        f = np.zeros((self.sim_params["N"], 2))
        shear_tmp = np.zeros((self.contacts["N"], 2))
        self.particles["f"] = f
        self.contacts["shear_tmp"] = shear_tmp

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
        dt = self.dt

        coords = self.particles["coords"]
        f = self.particles["f"]
        v = self.particles["v"]
        stiff = self.particles["stiffness"]
        r = self.particles["radius"]

        inds = self.contacts["inds"]
        ptrs = self.contacts["ptrs"]
        lookup = self.contacts["lookup"]
        prime_lookup = self.contacts["prime_lookup"]
        prime_IDs = self.contacts["prime_IDs"]
        offsets = self.contacts["offsets"]
        shear = self.contacts["shear"]
        N_all = len(lookup)

        # Temporary arrays to update contact shear
        temp = {
            "contact_normal": np.zeros_like(shear),
            "delta_shear": np.zeros_like(shear),
        }

        for n in range(N_all):
            neighbours = ptrs[inds[n]:inds[n+1]]
            i = lookup[n]
            x1, y1 = coords[i] + offsets[n]
            r1 = r[i]
            v1 = v[i]

            for k in neighbours:
                j = lookup[k]

                x2, y2 = coords[j] + offsets[k]
                r2 = r[j]
                v2 = v[j]

                d_sq = (x1 - x2)**2 + (y1 - y2)**2
                delta_sq = (r1 + r2)**2 - d_sq
                if delta_sq > 0:

                    # Identify contact
                    contact_id = prime_IDs[i] * prime_IDs[j]
                    contact_no = prime_lookup[contact_id]

                    # Extract shear history
                    shear_ij = shear[contact_no]

                    # Contact normal
                    inv_d = 1.0 / sqrt(d_sq)
                    nx = (x1 - x2)*inv_d
                    ny = (y1 - y2)*inv_d

                    """ Relative velocities (shear/normal) """

                    # Relative velocity
                    vrel = v1 - v2

                    # Normal component relative velocity
                    vn = vrel[0] * nx + vrel[1] * ny
                    vnx = vn * nx
                    vny = vn * ny

                    # Shear component relative velocity
                    vsx = vrel[0] - vnx
                    vsy = vrel[1] - vny

                    """ Contact forces """

                    # Particle overlap
                    delta = sqrt(delta_sq)

                    # Normal force
                    fn = stiff * delta
                    fnx = fn * nx
                    fny = fn * ny

                    # Shear force
                    fsx = -stiff * shear_ij[0]
                    fsy = -stiff * shear_ij[1]

                    """ Contact friction """

                    # Compute contact creep velocity
                    vc_ref = self.particles["vc_ref"]
                    inv_sinh_mu_ref = self.particles["inv_sinh_mu_ref"]
                    a_tilde = self.particles["a_tilde"]
                    inv_afn = 1.0 / (a_tilde * fn)
                    vcx = vc_ref * sinh(fsx * inv_afn) * inv_sinh_mu_ref
                    vcy = vc_ref * sinh(fsy * inv_afn) * inv_sinh_mu_ref

                    # Increment shear deficit
                    # Note factor 0.5 to prevent double counting
                    # Store contact shear in temporary array
                    temp["delta_shear"][contact_no][0] += 0.5 * (vsx - vcx) * dt
                    temp["delta_shear"][contact_no][1] += 0.5 * (vsy - vcy) * dt
                    # Store contact normal in temporary array
                    temp["contact_normal"][contact_no][0] = nx.copy()
                    temp["contact_normal"][contact_no][1] = ny.copy()

                    """ Particle forces """

                    # Increment particle forces
                    f[i][0] += fnx + fsx
                    f[i][1] += fny + fsy

        # Perform a second loop, this time over all contacts, and update
        # the contact shear deficit
        for i in range(len(shear)):

            # Contact normal components
            nx, ny = temp["contact_normal"][i]

            # Increment contact shear
            shear_ij = shear[i] + temp["delta_shear"][i]

            """ Rotate shear displacements onto contact plane """

            # Compute original shear magnitude
            shear_mag = shear_ij[0]**2 + shear_ij[1]**2

            # Compute normal component of shear vector
            shear_normal = shear_ij[0] * nx + shear_ij[1] * ny

            # Subtract normal component (gives tangential component)
            shear_ij[0] -= shear_normal * nx
            shear_ij[1] -= shear_normal * ny

            # Compute updated shear magnitude
            shear_mag_new = shear_ij[0]**2 + shear_ij[1]**2

            if shear_mag * shear_mag_new > 0:
                # Conserve shear magnitude
                shear_ratio = sqrt(shear_mag / shear_mag_new)
                shear_ij *= shear_ratio
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
        z_max = self.wall["x"]

        # Find particles near the top and bottom walls
        bottom_particles = (coords[:, 1] < r)
        top_particles = (coords[:, 1] > z_max - r)

        # Add particle-wall interactions
        f[bottom_particles, 1] += k*(r[bottom_particles] - coords[bottom_particles, 1])
        f_wall_top = k*(z_max - r[top_particles] - coords[top_particles, 1])
        f[top_particles, 1] += f_wall_top
        self.wall["f"] = -f_wall_top.sum()

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

