"""
Auxiliary functions to perform various (trivial) computations
"""

import numpy as np
from math import sqrt

class Elves:

    def __init__(self):
        pass

    def compute_dt_factor(self):
        """Compute (constant) time step factor"""
        m = self.particles["mass"].min()
        k = self.particles["stiffness"]
        safety_factor = 0.1
        self.dt = safety_factor * np.sqrt(m / k)
        # self.dt_factor = safety_factor * np.pi * sqrt((4.0/3.0)*np.pi * rho / K)

        pass

    @staticmethod
    def compute_reduced_mass(r1, r2):
        """Compute (equivalent) reduced mass"""
        return (r1*r2)**3 / (r1**3 + r2**3)

    def compute_dt_max(self):
        """
        Compute the maximum allowable time step (dt) for the current mesh
        """

        r = self.particles["radius"]
        self.dt = self.dt_factor * sqrt(r.min() ** 3)
        return self.dt

        N = self.sim_params["N"]
        inds = self.contacts["inds"]
        ptrs = self.contacts["ptrs"]
        r = self.particles["radius"]

        r12_min = r.max()

        for i in range(N):
            r1 = r[i]
            neighbours = ptrs[inds[i]:inds[i+1]]
            r2 = r[neighbours].min()
            r12 = self.compute_reduced_mass(r1, r2)
            if r12 < r12_min:
                r12_min = r12

        # self.dt = self.dt_factor * sqrt(r12_min)
        self.dt = self.dt_factor * sqrt(r.min()**3)
        return self.dt

    # https://rosettacode.org/wiki/Sieve_of_Eratosthenes#Infinite_generator_with_a_faster_algorithm
    def primes(self):
        yield 2
        yield 3
        yield 5
        yield 7
        bps = (p for p in self.primes())
        p = next(bps) and next(bps)
        q = p * p
        sieve = {}
        n = 9
        while True:
            if n not in sieve:
                if n < q:
                    yield n
                else:
                    p2 = p + p
                    sieve[q + p2] = p2
                    p = next(bps)
                    q = p * p
            else:
                s = sieve.pop(n)
                nxt = n + s
                while nxt in sieve:
                    nxt += s
                sieve[nxt] = s
            n += 2

