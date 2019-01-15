"""
Main DEM routine
"""
import copy

import numpy as np
from math import sinh

from src.artist import Rembrandt
from src.contact_detection import MeshDetection
from src.contact_model import ContactModel
from src.controller import WallControler
from src.helper import Elves
from src.integrator import Integrator
from src.writer import Tolstoy


class Fault2Dem(ContactModel, Elves, Integrator, MeshDetection,
                Rembrandt, Tolstoy, WallControler):

    def __init__(self):
        ContactModel.__init__(self)
        Elves.__init__(self)
        Integrator.__init__(self)
        MeshDetection.__init__(self)
        Rembrandt.__init__(self)
        Tolstoy.__init__(self)
        WallControler.__init__(self)
        pass

    def set_params(self, params):
        self.sim_params = {
            "N": params["N"],
            "box": params["box"],
            "damping_factor": params["damping_factor"],
            "gravity": params["gravity"],
            "drag": params["drag"],
            "v_lp": params["v_lp"],
        }
        self.output_params = {
            "screen_output": params["screen_output"],
            "file_output": params["file_output"],
        }
        self.user_params = params

        pass

    def init_domain(self):
        self.init_box()
        self.init_wall()
        self.init_sample()
        self.render_mesh()

        self.compute_dt_factor()
        # self.compute_dt_max()
        pass

    def init_box(self):
        pass

    def init_sample(self):

        params = self.user_params
        N = self.sim_params["N"]
        radii = np.random.uniform(params["r_min"], params["r_max"], size=N)
        m = (4.0 / 3.0) * np.pi * params["density"] * radii**3
        inv_m = 1.0 / np.vstack([m, m]).T

        particles = {
            "ID": np.arange(N),
            "r_min": params["r_min"],
            "r_max": params["r_max"],
            "density": params["density"],
            "stiffness": params["stiffness"],
            "Z_ps": params["Z_ps"],
            "mu": params["mu"],
            "radius": radii,
            "mass": m,
            "inv_mass": inv_m,
            "coords": np.zeros((N, 2)),
            "v": np.zeros((N, 2)),
            "f": np.zeros((N, 2)),
            "f_prev": np.zeros((N, 2)),
            "shear_dist": np.zeros(N),
        }

        particles["a_tilde"] = params["a_tilde"]
        particles["mu_ref"] = params["mu_ref"]
        particles["vc_ref"] = params["vc_ref"]

        i = 0
        fail_count = 0
        warning_issued = False

        while i < N:

            r = radii[i]
            x, y = np.random.uniform(0.01, 0.99, size=2)
            x = x*(self.sim_params["box"][0] - 2*r) + r
            y = y*(self.sim_params["box"][1] - 2*r) + r

            touched = False

            for j in range(i):
                x2, y2 = particles["coords"][j]
                r2 = particles["radius"][j]
                d_sq = (x - x2)**2 + (y - y2)**2
                if d_sq <= (r + r2)**2:
                    touched = True
                    fail_count += 1
                    break

            if not touched:
                particles["coords"][i] = (x, y)
                i += 1

            if (fail_count > 10*N) and not warning_issued:
                print("Warning: particle insertion failure count exceeded 10x N")
                print("Consider increasing domain size")
                print("Progress: \t %i / %i" % (i, N))
                warning_issued = True

        self.particles = particles

        return particles

    def run(self, steps):
        saved_data = []
        output = self.output_params
        self.screen_write_init()
        for i in range(steps):
            if i % output["screen_output"] == 0:
                self.screen_write()
            if i % output["file_output"] == 0:
                data = {
                    "particles": copy.deepcopy(self.particles),
                    "contacts": copy.deepcopy(self.contacts),
                }
                saved_data.append(data)
            self.do_step()
        self.saved_data = saved_data
        print()
        pass
