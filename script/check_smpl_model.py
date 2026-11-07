import os, sys
sys.path.insert(0, os.getcwd())
from uhc.smpllib.smpl_parser import SMPL_Parser
p = SMPL_Parser(model_path="data/smpl", gender="neutral", use_pca=False, create_transl=False)
print("SMPL model loaded OK")
for name in ["v_template", "faces", "J_regressor", "weights", "posedirs", "shapedirs", "parents"]:
    v = getattr(p, name, None)
    print(name, type(v).__name__, getattr(v, "shape", None))
