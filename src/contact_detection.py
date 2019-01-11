"""
Contact/collision detection algorithms and procedures
"""

import numpy as np
from scipy.spatial import Delaunay


class MeshDetection:

    def __init__(self):
        pass

    def render_mesh(self):
        """
        Perform a Delaunay triangulation to obtain the direct neighbours
        of each particle. Triangulation should be re-done when particle
        displacements exceed a certain threshold
        """

        points = self.particles["coords"]   # Particle baricentres
        r_max = self.particles["r_max"]
        x_max = self.sim_params["box"][0]
        IDs = self.particles["ID"]

        ghosts_left = np.where(points[:, 0] < r_max)[0]
        ghosts_right = np.where(points[:, 0] > x_max - r_max)[0]
        ghosts = np.hstack([ghosts_left, ghosts_right])

        N = self.sim_params["N"]
        N_left = len(ghosts_left)
        N_right = len(ghosts_right)
        offsets = np.zeros((N + N_left + N_right, 2))
        offsets[N:N+N_left, 0] = x_max
        offsets[N+N_left:, 0] = -x_max

        ghost_coords_left = points[ghosts_left]
        ghost_coords_right = points[ghosts_right]
        ghost_coords = np.vstack([ghost_coords_left, ghost_coords_right])

        all_IDs = np.hstack([IDs, ghosts])
        all_points = np.vstack([points, ghost_coords]) + offsets

        tri = Delaunay(all_points)              # Mesh object

        # Neighboring vertices of vertices. The indices of neighbouring
        # vertices of vertex k are obtained by ptrs[inds[k]:inds[k+1]]
        inds, ptrs = tri.vertex_neighbor_vertices

        # Store vertices and neighbours as contact dict
        self.contacts = {
            "inds": inds,
            "ptrs": ptrs,
            "points": tri.points.copy(),
            "lookup": all_IDs,
            "offsets": offsets,
        }

        return True

    def relocate_particles(self):
        points = self.particles["coords"]
        x_max = self.sim_params["box"][0]
        multiples = points[:, 0] // x_max
        points[:, 0] -= multiples * x_max
        pass


    def check_remesh(self):

        N = self.sim_params["N"]
        coords = self.particles["coords"]
        points = self.contacts["points"][:N]

        d = np.abs((coords - points).ravel())
        if d.max() > 0.5*self.particles["r_min"]:
            # print("remesh")
            self.relocate_particles()
            self.render_mesh()
