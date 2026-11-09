"""Apply / revert the layered-heightmap patch.

The building is a real stacked trimesh, so a single 2.5D heightfield cannot
describe it.  This patch keeps ONE global grid but stores `(B, nx, ny)` height
maps, one per terrain block, and resolves the block index per environment:

    heightsamples[px, py]   ->   heightsamples[block_id[env], px, py]

The block index lives on the MeshTerrain object, so the three sampling sites
(`get_heights`, `get_center_heights`, `_sample_static_terrain`) need no
signature changes -- they all go through `sample_height_points` / the terrain
object.

    python script/patch_layered_heightmap.py apply
    python script/patch_layered_heightmap.py status
    python script/patch_layered_heightmap.py revert
"""

import argparse
import os
import sys

TARGET = "pacer/env/tasks/humanoid_pedestrain_terrain.py"
BACKUP = TARGET + ".layered_bak"

# (old, new) -- `old` must occur exactly once, except for the indices listed in
# FIRST_ONLY, where only the first occurrence is replaced (MeshTerrain appears
# before the Terrain class, so the first match is always the mesh one).
FIRST_ONLY = {6, 7}
EDITS = [
    # ---- MeshTerrain: layered height samples + per-env block id -------------
    ("        self.heightsamples = torch.from_numpy(heigthmap_data['heigthmap']).to(device)",
     "        _hm = heigthmap_data['heigthmap']\n"
     "        if _hm.ndim == 2:\n"
     "            _hm = _hm[None]\n"
     "        self.heightsamples = torch.from_numpy(_hm).to(device)\n"
     "        self.num_blocks = int(self.heightsamples.shape[0])\n"
     "        self.block_id = torch.zeros(1, dtype=torch.long, device=device)"),

    ("        self.walkable_map  = torch.from_numpy(heigthmap_data['walkable_map']).to(device)",
     "        _wm = heigthmap_data['walkable_map']\n"
     "        if _wm.ndim == 2:\n"
     "            _wm = _wm[None]\n"
     "        self.walkable_map  = torch.from_numpy(_wm).to(device)"),

    ("        self.x_shape, self.y_shape = self.heightsamples.shape",
     "        self.x_shape, self.y_shape = (self.heightsamples.shape[1],\n"
     "                                      self.heightsamples.shape[2])"),

    ("        coord_x, coord_y = torch.where(self.walkable_map == 1) # Image coordinates, need to flip y and x",
     "        # every block shares one global grid, so the walkable set is the union\n"
     "        coord_x, coord_y = torch.where(self.walkable_map.sum(dim=0) > 0)"),

    ("    def sample_valid_locations(self, num_envs, env_ids):",
     "    def set_block_id(self, ids):\n"
     "        self.block_id = ids.to(self.device).long()\n"
     "\n"
     "    def block_rows(self, env_ids, batch):\n"
     "        if env_ids is None:\n"
     "            return self.block_id[:batch]\n"
     "        return self.block_id[env_ids]\n"
     "\n"
     "    def sample_valid_locations(self, num_envs, env_ids):"),

    # ---- sample_height_points: gather by block ------------------------------
    ("        heightsamples = self.heightsamples.clone()\n"
     "        device = points.device",
     "        heightsamples = self.heightsamples\n"
     "        _blk = self.block_rows(env_ids, B).repeat_interleave(N)\n"
     "        device = points.device"),

    ("            heightsamples_group = heightsamples[None, ].repeat(",
     "            heightsamples_group = heightsamples[0][None, ].repeat("),

    ("            heights1 = heightsamples[px, py]\n"
     "            heights2 = heightsamples[px + 1, py + 1]",
     "            heights1 = heightsamples[_blk, px, py]\n"
     "            heights2 = heightsamples[_blk, px + 1, py + 1]"),

    # ---- _sample_static_terrain: gather by block ----------------------------
    ("        heights1 = self.terrain.heightsamples[px, py]\n"
     "        heights2 = self.terrain.heightsamples[px + 1, py + 1]",
     "        _blk = self.terrain.block_rows(None, batch_size).repeat_interleave(num_points)\n"
     "        heights1 = self.terrain.heightsamples[_blk, px, py]\n"
     "        heights2 = self.terrain.heightsamples[_blk, px + 1, py + 1]"),

    # ---- create_mesh_ground: seed the block id from the scene ---------------
    ("        self.terrain = MeshTerrain(self.mesh_data, self.device)",
     "        self.terrain = MeshTerrain(self.mesh_data, self.device)\n"
     "        try:\n"
     "            _sb = int(self.mesh_data.get('start_block', 0))\n"
     "        except Exception:\n"
     "            _sb = 0\n"
     "        self.terrain.set_block_id(torch.full((self.num_envs,), _sb,\n"
     "                                             dtype=torch.long,\n"
     "                                             device=self.device))\n"
     "        self._block_id = self.terrain.block_id"),

    ("        self.height_samples = torch.tensor(self.terrain.heightsamples).to(",
     "        self.height_samples = torch.tensor(self.terrain.heightsamples[0]).to("),
    # ---- world_points_to_map: clip against the last two dims ----------------
    # shape[0]/[1] become (block, x) once the maps are layered; the pixel clip
    # must use the spatial dims or every sample lands in block 0 column 3.
    ("        px = torch.clip(px, 0, self.heightsamples.shape[0] -\n"
     "                        2)  # image, so sampling 1 is for x\n"
     "        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)",
     "        px = torch.clip(px, 0, self.heightsamples.shape[-2] -\n"
     "                        2)  # image, so sampling 1 is for x\n"
     "        py = torch.clip(py, 0, self.heightsamples.shape[-1] - 2)"),
    # ---- spawn snap: put the lowest body exactly on the surface -------------
    # The sampled reference pose rarely has its lowest body on the root ground
    # reference, so the spawn hovers a few cm and the character visibly drops
    # on frame 0.  Feet world z = root_z + (min body z - root ground ref).
    ("        root_pos[:, 2] += center_height\n"
     "        key_pos[..., 0:2] += diff_xy[:, None, :]",
     "        root_pos[:, 2] += center_height\n"
     "        if self.real_mesh:\n"
     "            _root_ref = root_pos[:, 2] - center_height\n"
     "            _delta = rb_pos[..., 2].min(dim=1).values - _root_ref\n"
     "            if float(_delta.abs().max()) < 1.5:\n"
     "                _new_root = center_height - _delta\n"
     "                _shift = _new_root - root_pos[:, 2]\n"
     "                root_pos[:, 2] = _new_root\n"
     "                rb_pos[..., 2] += _shift.unsqueeze(-1)\n"
     "                key_pos[..., 2] += _shift[:, None]\n"
     "        key_pos[..., 0:2] += diff_xy[:, None, :]"),
]


def detect_nl(src):
    return "\r\n" if "\r\n" in src else "\n"


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def apply(root):
    path = os.path.join(root, TARGET)
    src = read(path)
    nl = detect_nl(src)
    if "layered height samples" in src or "self.num_blocks = int(" in src:
        print("already applied")
        return 0
    if not os.path.exists(os.path.join(root, BACKUP)):
        write(os.path.join(root, BACKUP), src)
        print("backup ->", BACKUP)

    for i, (old, new) in enumerate(EDITS):
        o = old.replace("\n", nl)
        n = new.replace("\n", nl)
        cnt = src.count(o)
        if i in FIRST_ONLY:
            if cnt < 1:
                print("EDIT %d: anchor missing\n  %r" % (i, old[:80]))
                return 1
            src = src.replace(o, n, 1)
            continue
        if cnt != 1:
            print("EDIT %d: anchor count = %d (expected 1)\n  %r" % (i, cnt, old[:80]))
            return 1
        src = src.replace(o, n)
    write(path, src)
    print("applied %d edits to %s" % (len(EDITS), TARGET))
    return 0


def revert(root):
    bak = os.path.join(root, BACKUP)
    if not os.path.exists(bak):
        print("no backup to revert from")
        return 1
    write(os.path.join(root, TARGET), read(bak))
    os.remove(bak)
    print("reverted", TARGET)
    return 0


def status(root):
    path = os.path.join(root, TARGET)
    src = read(path)
    marks = [("num_blocks", "self.num_blocks = int(" in src),
             ("block_id tensor", "def set_block_id" in src),
             ("gather in sample_height_points", "heightsamples[_blk, px, py]" in src),
             ("gather in _sample_static_terrain", "self.terrain.heightsamples[_blk, px, py]" in src),
             ("start_block seed", "self._block_id = self.terrain.block_id" in src)]
    for name, ok in marks:
        print("  %-34s %s" % (name, "OK" if ok else "MISSING"))
    print("  backup present:", os.path.exists(os.path.join(root, BACKUP)))
    return 0 if all(v for _, v in marks) else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["apply", "revert", "status"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    return {"apply": apply, "revert": revert, "status": status}[a.action](a.root)


if __name__ == "__main__":
    sys.exit(main())
