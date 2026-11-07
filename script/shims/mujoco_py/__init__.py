"""Minimal mujoco_py shim for the offline AMASS conversion path.

The conversion only needs SMPL_BONE_ORDER_NAMES from uhc.smpllib.smpl_mujoco;
that module imports mujoco_py at import time even though the converter never
calls the MuJoCo runtime.  This shim keeps the import side-effect-free without
touching PACER source files.
"""


def load_model_from_path(*args, **kwargs):
    raise RuntimeError("mujoco_py shim: runtime MuJoCo is not available")


class MjSim:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("mujoco_py shim: MjSim is not available")


class MjRenderContextOffscreen:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("mujoco_py shim: MjRenderContextOffscreen is not available")
