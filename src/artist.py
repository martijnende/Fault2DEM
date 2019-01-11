"""
Collection of function related to visualisation and animation
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Rembrandt:

    def __init__(self):
        pass

    def draw_scene(self):

        N = self.sim_params["N"]
        N_ghosts = len(self.contacts["lookup"]) - N
        x_max = self.sim_params["box"][0]
        particles = self.particles
        lookup = self.contacts["lookup"]
        offsets = self.contacts["offsets"]

        ax = plt.subplot(111, aspect="equal")
        plt.xlim((0, self.sim_params["box"][0]))
        plt.ylim((0, self.sim_params["box"][1]))
        for i in range(N):
            x, y = particles["coords"][i]
            r = particles["radius"][i]

            circle = plt.Circle((x, y), r, color="C0", alpha=0.5)
            plt.text(x, y, "%i" % i, va="center", ha="center")
            ax.add_artist(circle)

        for n in range(N_ghosts):
            i = lookup[N+n]
            x, y = particles["coords"][i] + offsets[N+n]
            r = particles["radius"][i]

            circle = plt.Circle((x, y), r, color="C0", alpha=0.5)
            ax.add_artist(circle)

        plt.tight_layout()
        plt.show()

    def init_animation(self):
        return []

    def update_canvas(self, n):

        N = self.sim_params["N"]
        N_ghosts = len(self.contacts["lookup"]) - N
        particles = self.saved_data[n]["particles"]
        lookup = self.contacts["lookup"]
        offsets = self.contacts["offsets"]

        self.canvas.cla()
        self.canvas.set_aspect("equal")
        self.canvas.set_xlim((0, self.sim_params["box"][0]))
        self.canvas.set_ylim((0, self.sim_params["box"][1]))

        for i in range(N):
            x, y = particles["coords"][i]
            r = particles["radius"][i]

            circle = plt.Circle((x, y), r, color="C0", alpha=0.5)
            self.canvas.add_artist(circle)

        for n in range(N_ghosts):
            i = lookup[N+n]
            x, y = particles["coords"][i] + offsets[N+n]
            r = particles["radius"][i]

            circle = plt.Circle((x, y), r, color="C0", alpha=0.5)
            self.canvas.add_artist(circle)

        return []

    def animate(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.set_aspect("equal")
        ax.set_xlim((0, self.sim_params["box"][0]))
        ax.set_ylim((0, self.sim_params["box"][1]))
        self.canvas = ax
        T = np.arange(1, len(self.saved_data))

        ani = animation.FuncAnimation(
            fig, self.update_canvas, init_func=self.init_animation,
            frames=T, interval=20, blit=True)

        plt.show()
        pass

