"""Apply / revert the multi-surface height-query patch.

A single 2.5D heightfield cannot describe stacked floors, but the controller
never needs "which floor am I on" -- it only needs, for every cell in its 4x4 m
field of view, the height of the surface it could actually stand on:

    h(x, y) = max { surface_s(x, y) : surface_s(x, y) <= foot_z + RISE_TOL }

So the data is a stack of surface sheets (one per physical surface: each slab
and each flight) and the query is per cell, resolved by the agent's own foot
height.  No block id, no switching, no debounce, no portals: the whole
state machine disappears and the field of view simply follows the geometry as
the agent walks.

    python script/patch_multi_surface_query.py apply
    python script/patch_multi_surface_query.py status
    python script/patch_multi_surface_query.py revert
"""

import argparse
import os
import sys

T_TARGET = "pacer/env/tasks/humanoid_pedestrain_terrain.py"
T_BACKUP = T_TARGET + ".multisurf_bak"

FIRST_ONLY = {7, 8}
REPLACE_ALL = {15}

EDITS = [
    # ---- MeshTerrain: a stack of surface sheets -----------------------------
    ("        self.heightsamples = torch.from_numpy(heigthmap_data['heigthmap']).to(device)",
     "        _hm = heigthmap_data['heigthmap']\n"
     "        if _hm.ndim == 2:\n"
     "            _hm = _hm[None]\n"
     "        self.heightsamples = torch.from_numpy(_hm).to(device)\n"
     "        self.num_surfaces = int(self.heightsamples.shape[0])\n"
     "        self.num_blocks = self.num_surfaces\n"
     "        # how far above the foot a surface may sit and still be walkable\n"
     "        self.rise_tol = float(heigthmap_data.get('rise_tol', 0.35))"),

    ("        self.walkable_map  = torch.from_numpy(heigthmap_data['walkable_map']).to(device)",
     "        _wm = heigthmap_data['walkable_map']\n"
     "        if _wm.ndim == 2:\n"
     "            _wm = _wm[None]\n"
     "        self.walkable_map  = torch.from_numpy(_wm).to(device)"),

    ("        self.x_shape, self.y_shape = self.heightsamples.shape",
     "        self.x_shape, self.y_shape = (self.heightsamples.shape[1],\n"
     "                                      self.heightsamples.shape[2])"),

    ("        coord_x, coord_y = torch.where(self.walkable_map == 1) # Image coordinates, need to flip y and x",
     "        # every sheet shares one global grid, so the walkable set is the union\n"
     "        coord_x, coord_y = torch.where(self.walkable_map.sum(dim=0) > 0)"),

    ("    def sample_valid_locations(self, num_envs, env_ids):",
     "    def surface_height(self, px, py, foot_z):\n"
     "        \"\"\"Highest sheet at (px, py) that is not above foot_z + rise_tol.\n"
     "\n"
     "        px/py are flat index tensors of length B*N; foot_z is (B,).  Points\n"
     "        with no usable sheet fall back to PIT (a void).\n"
     "        \"\"\"\n"
     "        B = foot_z.shape[0]\n"
     "        surf = self.heightsamples[:, px, py]\n"
     "        N = py.numel() // max(B, 1)\n"
     "        surf = surf.view(self.num_surfaces, B, N).permute(1, 2, 0)\n"
     "        lim = (foot_z + self.rise_tol).view(B, 1, 1)\n"
     "        ok = surf <= lim\n"
     "        void = torch.full_like(surf, float('-1.5'))\n"
     "        return torch.where(ok, surf, void).max(dim=-1).values\n"
     "\n"
     "    def sample_valid_locations(self, num_envs, env_ids):"),

    ("    def sample_height_points(self,\n"
     "                             points,",
     "    def sample_height_points(self,\n"
     "                             points,\n"
     "                             foot_z=None,"),

    # ---- sample_height_points: per-cell surface query -----------------------
    ("        heightsamples = self.heightsamples.clone()\n"
     "        device = points.device",
     "        device = points.device\n"
     "        _foot = foot_z if foot_z is not None else torch.full(\n"
     "            (B,), 1e3, device=points.device, dtype=points.dtype)"),

    ("            heightsamples_group = heightsamples[None, ].repeat(",
     "            heightsamples_group = self.heightsamples[0][None, ].repeat("),

    ("            heights1 = heightsamples[px, py]\n"
     "            heights2 = heightsamples[px + 1, py + 1]\n"
     "            heights = torch.min(heights1, heights2)",
     "            heights = torch.minimum(\n"
     "                self.surface_height(px, py, _foot),\n"
     "                self.surface_height(px + 1, py + 1, _foot))"),

    # ---- _sample_static_terrain: per-cell surface query ---------------------
    ("        heights1 = self.terrain.heightsamples[px, py]\n"
     "        heights2 = self.terrain.heightsamples[px + 1, py + 1]",
     "        _ref = self._surface_ref(batch_size)\n"
     "        heights1 = self.terrain.surface_height(px, py, _ref)\n"
     "        heights2 = self.terrain.surface_height(px + 1, py + 1, _ref)"),

    # ---- create_mesh_ground: no block seeding, just the spawn reference -----
    ("        self.terrain = MeshTerrain(self.mesh_data, self.device)",
     "        self.terrain = MeshTerrain(self.mesh_data, self.device)\n"
     "        try:\n"
     "            self._spawn_ref_z = float(self.mesh_data.get('start_z', 1e3))\n"
     "        except Exception:\n"
     "            self._spawn_ref_z = 1e3\n"
     "        self._block_paths = self.mesh_data.get('block_paths')\n"
     "        self._block_id = None"),

    ("        self.height_samples = torch.tensor(self.terrain.heightsamples).to(",
     "        self.height_samples = torch.tensor(self.terrain.heightsamples.max(dim=0).values).to("),

    # ---- world_points_to_map: clip the last two dims ------------------------
    ("        px = torch.clip(px, 0, self.heightsamples.shape[0] -\n"
     "                        2)  # image, so sampling 1 is for x\n"
     "        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)",
     "        px = torch.clip(px, 0, self.heightsamples.shape[-2] -\n"
     "                        2)  # image, so sampling 1 is for x\n"
     "        py = torch.clip(py, 0, self.heightsamples.shape[-1] - 2)"),

    # ---- spawn snap: feet exactly on the surface ---------------------------
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

    # ---- per-step foot reference + obs hook --------------------------------
    ("    def create_mesh_ground(self):",
     "    def _surface_ref(self, batch):\n"
     "        \"\"\"Reference foot height used to pick the walkable sheet.\"\"\"\n"
     "        ref = getattr(self, \"_foot_ref\", None)\n"
     "        # a fresh episode must resolve against the scene start height: the\n"
     "        # leftover foot reference from the previous episode belongs to another\n"
     "        # pose and would spawn the character at the wrong level.\n"
     "        _pb = getattr(self, \"progress_buf\", None)\n"
     "        _fresh = _pb is not None and int(_pb.min()) <= 1\n"
     "        if ref is None or _fresh or ref.shape[0] < batch:\n"
     "            z0 = float(getattr(self, \"_spawn_ref_z\", 1e3))\n"
     "            return torch.full((batch,), z0, device=self.device)\n"
     "        return ref[:batch].to(self.device)\n"
     "\n"
     "    def _compute_observations(self, env_ids=None):\n"
     "        # the field of view is resolved against the current foot height, so\n"
     "        # refresh it once per step before the heightmap observation is built\n"
     "        with torch.no_grad():\n"
     "            self._foot_ref = self._rigid_body_pos[:, :, 2].min(dim=1).values\n"
     "        return super()._compute_observations(env_ids)\n"
     "\n"
     "    def create_mesh_ground(self):"),

    # ---- pass the foot reference into the terrain query (both call sites) --
    ("self.terrain.sample_height_points(points.clone(), env_ids=env_ids)",
     "self.terrain.sample_height_points(points.clone(), env_ids=env_ids, foot_z=self._surface_ref(points.shape[0]))"),
    # ---- a respawn must resolve against the scene start height --------------
    # The leftover foot reference belongs to the pose the previous episode
    # ended in, so without clearing it the character respawns at the wrong
    # level (3F spot, 2F height) and immediately falls through.
    ("    def _reset_ref_state_init(self, env_ids):",
     "    def _reset_ref_state_init(self, env_ids):\n"
     "        self._foot_ref = None"),
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
    path, backup = os.path.join(root, T_TARGET), os.path.join(root, T_BACKUP)
    src = read(path)
    nl = detect_nl(src)
    if "self.num_surfaces = int(" in src:
        print("already applied")
        return 0
    if not os.path.exists(backup):
        write(backup, src)
        print("backup ->", os.path.basename(backup))
    for i, (old, new) in enumerate(EDITS):
        o, n = old.replace("\n", nl), new.replace("\n", nl)
        cnt = src.count(o)
        if i in REPLACE_ALL:
            if cnt < 1:
                print("EDIT %d: anchor missing\n  %r" % (i, old[:80]))
                return 1
            src = src.replace(o, n)
            continue
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
    print("applied %d edits" % len(EDITS))
    return 0


def revert(root):
    path, backup = os.path.join(root, T_TARGET), os.path.join(root, T_BACKUP)
    if not os.path.exists(backup):
        print("no backup")
        return 1
    write(path, read(backup))
    os.remove(backup)
    print("reverted", T_TARGET)
    return 0


def status(root):
    src = read(os.path.join(root, T_TARGET))
    marks = [("surface stack", "self.num_surfaces = int(" in src),
             ("query method", "def surface_height" in src),
             ("obs hook", "self._foot_ref = self._rigid_body_pos" in src),
             ("no block machine", "def _update_block_id" not in src),
             ("clip fixed", "shape[-2]" in src)]
    for name, ok in marks:
        print("  %-24s %s" % (name, "+" if ok else "-"))
    return 0 if all(v for _, v in marks) else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["apply", "revert", "status"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    return {"apply": apply, "revert": revert, "status": status}[a.action](a.root)


if __name__ == "__main__":
    sys.exit(main())
