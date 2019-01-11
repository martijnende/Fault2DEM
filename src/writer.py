"""
Simulation output routines
"""


class Tolstoy:

    def __init__(self):
        pass

    @staticmethod
    def screen_write_init():
        print("step \t time \t wall x \t wall f \t wall f diff \t E_tot")

    def screen_write(self):
        wall_diff = 100 * (self.wall["f_target"] - self.wall["f"]) / self.wall["f_target"]
        E_tot = self.compute_energy()
        items = (self.step, self.t, self.wall["x"], self.wall["f"], wall_diff, E_tot)
        print("%i \t %.3e \t %.2f \t %.2e \t %.2f \t %.2e" % items)
