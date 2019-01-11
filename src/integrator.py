"""
Time/position integration routine
"""


class Integrator:

    step = 0
    t = 0.0

    def __init__(self):
        pass

    def do_step(self):

        self.verlet_integration("x")
        self.update_forces()
        self.verlet_integration("v")
        self.move_wall()
        # self.compute_energy()
        self.check_remesh()
        self.step += 1
        self.t += self.dt

        pass

    def verlet_integration(self, which):

        dt = self.dt
        x = self.particles["coords"]
        v = self.particles["v"]
        f = self.particles["f"]
        f_prev = self.particles["f_prev"]
        inv_m = self.particles["inv_mass"]

        if which == "x":
            x += v * dt + 0.5 * inv_m * f * dt**2
            self.particles["f_prev"] = f
        elif which == "v":
            v += 0.5 * inv_m * (f + f_prev) * dt

        pass

    def compute_energy(self):

        x = self.particles["coords"]
        v = self.particles["v"]
        m = self.particles["mass"]

        E_kin = 0.5*m*v[:, 1]**2
        E_pot = 9.81*m*x[:, 1]

        return (E_kin + E_pot).sum()


