import pandas as pd, numpy as np
from src.safety import OODProfile

def test_ood_profile():
    d=pd.DataFrame({"part_id":[f"P{i}" for i in range(30)],"component_family":["F"]*30,"a__current":np.random.default_rng(1).normal(size=30),"a__robust_z":np.random.default_rng(2).normal(size=30),"future_defective":[0]*30})
    p=OODProfile().fit(d); o=p.score(d); assert set(o.ood_status).issubset({"LOW","MODERATE","SEVERE"})
