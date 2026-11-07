class _Cymj:
    class MjRenderContextWindow:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("mujoco_py shim: MjRenderContextWindow is not available")


cymj = _Cymj()
