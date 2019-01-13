"""
Contact/collision detection algorithms and procedures
"""

import numpy as np
from scipy.spatial import Delaunay
from itertools import islice


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

        # Identify candidates for ghost particles
        ghosts_left = np.where(points[:, 0] < r_max)[0]
        ghosts_right = np.where(points[:, 0] > x_max - r_max)[0]
        ghosts = np.hstack([ghosts_left, ghosts_right])

        # Count real/ghost particles
        N = self.sim_params["N"]
        N_left = len(ghosts_left)
        N_right = len(ghosts_right)

        # Calculate position offset of ghosts
        offsets = np.zeros((N + N_left + N_right, 2))
        offsets[N:N+N_left, 0] = x_max
        offsets[N+N_left:, 0] = -x_max

        # Collect ghost praticle coordinates
        ghost_coords_left = points[ghosts_left]
        ghost_coords_right = points[ghosts_right]
        ghost_coords = np.vstack([ghost_coords_left, ghost_coords_right])

        # Generate prime numbered IDs for particles
        prime_IDs = np.array(list(islice(self.primes(), N)), dtype=int)
        # Create lookup table for real/ghost particles
        all_IDs = np.hstack([IDs, ghosts])
        # Particle centroids used in triangulation (including offsets)
        all_points = np.vstack([points, ghost_coords]) + offsets

        # Create mesh object
        tri = Delaunay(all_points)

        # Neighboring vertices of vertices. The indices of neighbouring
        # vertices of vertex k are obtained by ptrs[inds[k]:inds[k+1]]
        inds, ptrs = tri.vertex_neighbor_vertices

        # Create lookup table for particle contacts
        prime_lookup = {}
        # Loop over all particles (including ghosts)
        for n in range(len(all_IDs)):
            # Get the neighbours of this particle
            neighbours = ptrs[inds[n]:inds[n + 1]]
            i = all_IDs[n]
            # Loop over all particle neighbours
            for k in neighbours:
                j = all_IDs[k]
                # Generate a particle-pair ID from primes
                contact_id = prime_IDs[i] * prime_IDs[j]
                # Create dictionary entry
                prime_lookup[contact_id] = 0

        # Number of particle-pair contacts
        N_contacts = inds[-1] // 2
        # Contact IDs
        keys = sorted(prime_lookup.keys())
        # Create new lookup table for contacts
        prime_lookup = dict(zip(keys, np.arange(N_contacts)))

        # Store vertices and neighbours as contact dict
        self.contacts = {
            "N": N_contacts,
            "inds": inds,
            "ptrs": ptrs,
            "points": tri.points.copy(),
            "lookup": all_IDs,
            "offsets": offsets,
            "prime_lookup": prime_lookup,
            "prime_IDs": prime_IDs,
            "shear": np.zeros((N_contacts, 2)),
            "forces": np.zeros((N_contacts, 2)),
            "tri": tri,
        }

        return True

    def relocate_particles(self):
        """Relocate particle positions within simulation domain"""
        points = self.particles["coords"]
        x_max = self.sim_params["box"][0]
        # If the particle position is a multiple of the (periodic)
        # simulation domain: subtract multiplier value
        multiples = points[:, 0] // x_max
        points[:, 0] -= multiples * x_max
        pass


    def check_remesh(self):
        """Check if particle displacements demand remeshing"""

        N = self.sim_params["N"]
        coords = self.particles["coords"]
        points = self.contacts["points"][:N]

        d = np.abs((coords - points).ravel())
        if d.max() > 0.25*self.particles["r_min"]:
            # print("remesh")
            self.relocate_particles()
            self.write_history()
            self.render_mesh()
            self.rewrite_history()

    def write_history(self):
        """Store shear history values prior to remeshing"""
        self.lookup_history = self.contacts["prime_lookup"].copy()
        self.shear_history = self.contacts["shear"].copy()

    def rewrite_history(self):
        """Restore shear history values after remeshing"""

        # Current lookup dictionary and shear values
        lookup = self.contacts["prime_lookup"]
        shear = self.contacts["shear"]

        # Previous lookup dictionary and shear values
        lookup_history = self.lookup_history
        shear_history = self.shear_history
        old_keys = lookup_history.keys()

        # Loop over all current contacts
        for key, val in lookup.items():
            # If contact existed before:
            if key in old_keys:
                # Get array index for contact shear history
                old_key = lookup_history[key]
                # Copy shear history into new contact list
                shear[val] = shear_history[old_key]

        pass
