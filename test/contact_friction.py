"""
Test suite for the contact friction model
"""

import numpy as np
from src.contact_model import ContactModel
from src.contact_detection import MeshDetection

"""
1) Set-up sample with 1 particle contact
2) Perform triangulation
3) Enforce motion + remeshing

Tests:
- mu_ss = mu_ref + a*log(v/v_ref)
- reverse motion
- independent of normal force
- shear rotation onto contact plane
- no shear change when v_rel = vn (vs = 0)
- remeshing ok?
"""
