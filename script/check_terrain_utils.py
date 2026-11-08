"""Check which terrain helpers isaacgym exposes in this install."""
import inspect

from isaacgym import terrain_utils as tu

print([n for n in dir(tu) if not n.startswith("_")])
print(inspect.signature(tu.convert_heightfield_to_trimesh))
src = inspect.getsource(tu.convert_heightfield_to_trimesh)
print(src[:700])
