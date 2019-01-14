"""
Controller class for top wall position
"""
import numpy as np


class WallControler:

    def __init__(self):
        pass

    def init_wall(self):

        # WARNING: re-run with different params does not update wall params!

        self.wall = {
            "x": self.sim_params["box"][1],
            "v": 0.0,
            "f": 0.0,
            "k": self.user_params["servo_k"] / self.user_params["f_target"],
            "f_target": self.user_params["f_target"],
        }
        self.wall["k"] *= self.wall["f_target"]

        pass

    def move_wall(self):

        f_target = self.wall["f_target"]
        f = self.wall["f"]
        self.wall["x"] += self.wall["k"] * (f - f_target) * self.dt
        # print(self.wall["x"], f)
