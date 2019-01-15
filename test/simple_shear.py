from src.main import Fault2Dem
import numpy as np

params = {
    "N": 150,
    "box": [30, 20],
    "damping_factor": 0.1,
    "gravity": 0.0,
    "drag": 0,
    "v_lp": 0,
    "screen_output": 100,
    "file_output": 100,
    "r_min": 0.6,
    'r_max': 0.8,
    "density": 2500,
    "stiffness": 1e7,
    "a_tilde": 0.005,
    "mu_ref": 0.6,
    "vc_ref": 1.0,
    "servo_k": 1e-5,
    "f_target": 1e6,
    "Z_ps": 1e-11,
    "mu": 0.6
}

np.random.seed(0)
DEM = Fault2Dem()
DEM.set_params(params)
DEM.init_domain()
# DEM.draw_scene(tri=True)
DEM.run(5000)
# DEM.draw_scene(tri=True)
# DEM.animate(tri=True)

params["damping_factor"] = 0.0
params["v_lp"] = 1e0
params["drag"] = 1e6
DEM.set_params(params)
DEM.run(10000)

params["v_lp"] = 1e1
DEM.set_params(params)
DEM.run(5000)

params["v_lp"] = 1e0
DEM.set_params(params)
DEM.run(10000)


DEM.animate(tri=True)
