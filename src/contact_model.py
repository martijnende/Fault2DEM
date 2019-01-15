"""

"""

import numpy as np
from math import sqrt, sinh, exp, log, tanh, atanh


class ContactModel:

    def __init__(self):
        pass

    def init_step(self):
        """ Initialise step """
        f = np.zeros((self.sim_params["N"], 2))
        self.particles["f"] = f
        pass

    def update_forces(self):
        self.init_step()
        self.update_wall_forces()
        self.update_body_forces()
        self.update_particle_forces()
        self.add_damping()
        pass

    def update_particle_forces(self):
        """ Compute particle-particle interaction forces """

        dt = self.dt

        # Particle related quantities
        coords = self.particles["coords"]
        offsets = self.contacts["offsets"]
        r = self.particles["radius"]
        f = self.particles["f"]
        v = self.particles["v"]
        stiff = self.particles["stiffness"]
        Z_ps = self.particles["Z_ps"]
        mu = self.particles["mu"]

        # Contact related quantities
        shear = self.contacts["shear"]

        # Bookkeeping arrays
        inds = self.contacts["inds"]
        ptrs = self.contacts["ptrs"]
        lookup = self.contacts["lookup"]
        prime_lookup = self.contacts["prime_lookup"]
        prime_IDs = self.contacts["prime_IDs"]

        N_all = len(lookup)

        # Loop over all particles (including ghosts)
        # Note that this visits each contact twice
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
                    prime_i = prime_IDs[i]
                    prime_j = prime_IDs[j]
                    contact_id = prime_i * prime_j
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

                    # Dissolved amount
                    delta_ps = shear_ij[2]

                    # Normal force
                    fn = stiff * (delta - delta_ps)
                    fnx = fn * nx
                    fny = fn * ny

                    # Shear force
                    fsx = -stiff * shear_ij[0]
                    fsy = -stiff * shear_ij[1]
                    # Change sign of fs for other contact

                    """ Contact friction """

                    # Compute/store contact friction for smallest particle ID

                    if prime_i < prime_j:

                        # Tangent creep direction (parallel to shear force)
                        fs = sqrt(fsx**2 + fsy**2)
                        s_ratio = 1.0

                        # Problem combination IPS/contact friction?
                        if (fs > 10) and (fn > 10):
                            # Compute contact creep velocity based on Chen's
                            # friction model. A purely forward approach is
                            # highly unstable owing to the sinh/exponential
                            # dependence on stress. Instead, the shear creep
                            # (relaxation) is computed incrementally from an
                            # analytical solution for:
                            #   ds/dx = - vc_ref * exp([mu(s) - mu_ref] / a)
                            # The new value of s(t + dt) is computed, and the
                            # shear_ij vector is "shortened" by a ratio
                            # s(t + dt) / s(t). Then, the contribution from
                            # elastic shear loading (vs * dt) is added.

                            vc_ref = self.particles["vc_ref"]
                            mu_ref = self.particles["mu_ref"]
                            a_tilde = self.particles["a_tilde"]

                            # Current shear deficit
                            z0 = exp(- (fs / fn - mu_ref) / a_tilde)
                            z = stiff * vc_ref * dt / (a_tilde * fn)

                            # Shear deficit after relaxation by creep
                            s0 = fs / stiff
                            s = (fn / stiff) * (mu_ref - a_tilde*np.log(z + z0))
                            if s0 < s:
                                s_ratio = 1
                            else:
                                s_ratio = s / s0

                        shear_ij[0] = shear_ij[0] * s_ratio + vsx * dt
                        shear_ij[1] = shear_ij[1] * s_ratio + vsy * dt

                        # shear_ij[0] += vsx * dt
                        # shear_ij[1] += vsy * dt
                        #
                        # fsx = -stiff * shear_ij[0]
                        # fsy = -stiff * shear_ij[1]
                        # fs = sqrt(fsx**2 + fsy**2)
                        #
                        # if fs > mu * fn:
                        #     if fs != 0.0:
                        #         f_ratio = mu * fn / fs
                        #         shear_ij[0] *= f_ratio
                        #         shear_ij[1] *= f_ratio
                        #         fsx *= f_ratio
                        #         fsy *= f_ratio
                        #         # fs *= f_ratio

                        self.contacts["forces"][contact_no][0] = fs
                        self.contacts["forces"][contact_no][1] = fn

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
                            shear_ij[0] *= shear_ratio
                            shear_ij[1] *= shear_ratio
                    # Change sign of shear force for "other" particle
                    else:
                        fsx *= -1
                        fsy *= -1

                    """ Pressure solution """

                    if prime_i < prime_j:

                        if fn * Z_ps > 0:

                            # Contact plane radius
                            r1_sq = r1**2
                            r2_sq = r2**2
                            rc_sq = r1_sq - 1.0 / (4.0 * d_sq) * (d_sq - r2_sq + r1_sq)**2

                            # Pressure solution rate
                            PS_rate = 2.0 * Z_ps * fn / (np.pi * rc_sq**2)
                            if PS_rate * dt >= delta:
                                shear_ij[2] = 0.0
                            else:
                                shear_ij[2] += PS_rate * dt

                    """ Particle forces """

                    # Increment particle forces
                    f[i][0] += fnx + fsx
                    f[i][1] += fny + fsy
                    # self.contacts["shear"][contact_no] = shear_ij
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
        y_min = coords[:, 1].min()
        y_med = 0.5*(y_max + y_min)
        height = y_max - y_min
        r_max = self.particles["r_max"]

        # Gravitational pull
        if np.abs(self.sim_params["gravity"]) > 0:
            f[:, 1] += self.sim_params["gravity"] * m

        # Shear drag
        if self.sim_params["drag"] * self.sim_params["v_lp"] > 0:
            # Compute position-based load-point velocity
            # (zero at middle of the sample)
            v_lp = self.sim_params["v_lp"]
            v_lp_y = v_lp * (coords[:, 1] - y_med) / height
            dx += (v_lp_y - v[:, 0]) * self.dt

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

