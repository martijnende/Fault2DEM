from src.main import Fault2Dem
import numpy as np

params = {
    "N": 10,
    "box": [10, 10],
    "damping_factor": 0.1,
    "gravity": -9.81,
    "drag": 0,
    "strain_rate": 0,
    "screen_output": 10,
    "file_output": 10,
    "r_min": 0.6,
    'r_max': 0.8,
    "density": 2500,
    "stiffness": 1e7,
    "a_tilde": 0.005,
    "mu_ref": 0.6,
    "vc_ref": 1.0,
    "servo_k": 0.0,
    "f_target": 1e6,
}

np.random.seed(0)
DEM = Fault2Dem()
DEM.set_params(params)
DEM.init_domain()
DEM.draw_scene(tri=True)
DEM.run(570)
DEM.draw_scene(tri=True)
# DEM.animate(tri=False)
