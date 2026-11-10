"""Apply / revert the layered-heightmap + block-state-machine patch.

The building is a real stacked trimesh, so a single 2.5D heightfield cannot
describe it.  This patch keeps ONE global grid but stores `(B, nx, ny)` height
maps, one per terrain block, and resolves the block index per environment:

    heightsamples[px, py]   ->   heightsamples[block_id[env], px, py]

The block index lives on the MeshTerrain object, so the three sampling sites
(`get_heights`, `get_center_heights`, `_sample_static_terrain`) need no
signature changes -- they all go through `sample_height_points` / the terrain
object.  `_update_block_id()` assigns the block from the landing surface plus
the motion direction (design doc 3.3) and runs once per sim step from
`_compute_observations`.

    python script/patch_layered_heightmap.py apply
    python script/patch_layered_heightmap.py status
    python script/patch_layered_heightmap.py revert
"""

import argparse
import os
import sys

T_TARGET = "pacer/env/tasks/humanoid_pedestrain_terrain.py"
T_BACKUP = T_TARGET + ".layered_bak"
B_TARGET = "pacer/env/tasks/base_task.py"
B_BACKUP = B_TARGET + ".layered_bak"

# (old, new) for the terrain task.  `old` must occur exactly once, except for
# the indices listed in FIRST_ONLY where only the first occurrence is replaced
# (MeshTerrain appears before the Terrain class, so the first match is the mesh
# one).
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
     "        self._block_id = self.terrain.block_id\n"
     "        try:\n"
     "            self._scene_start_yaw = float(self.mesh_data.get(\"start_yaw\", 0.0))\n"
     "        except Exception:\n"
     "            self._scene_start_yaw = 0.0"),

    ("        self.height_samples = torch.tensor(self.terrain.heightsamples).to(",
     "        self.height_samples = torch.tensor(self.terrain.heightsamples[0]).to("),

    # ---- world_points_to_map: clip against the last two dims ----------------
    ("        px = torch.clip(px, 0, self.heightsamples.shape[0] -\n"
     "                        2)  # image, so sampling 1 is for x\n"
     "        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)",
     "        px = torch.clip(px, 0, self.heightsamples.shape[-2] -\n"
     "                        2)  # image, so sampling 1 is for x\n"
     "        py = torch.clip(py, 0, self.heightsamples.shape[-1] - 2)"),

    # ---- spawn snap: put the lowest body exactly on the surface -------------
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

    # ---- block state machine (design doc 3.3) ------------------------------
    ("    def create_mesh_ground(self):",
     "    def _update_block_id(self):\n"
     "        \"\"\"Landing surface first, motion direction breaks ties.\n"
     "\n"
     "        Candidates are the blocks whose surface is within 15 cm of the\n"
     "        lowest body point.  At a portal both the floor and the stair\n"
     "        qualify and their heights are identical, so the direction only\n"
     "        decides which map to read next -- judging wrong there costs\n"
     "        nothing.\n"
     "        \"\"\"\n"
     "        if not getattr(self, \"real_mesh\", False):\n"
     "            return\n"
     "        if getattr(self.terrain, \"num_blocks\", 1) <= 1:\n"
     "            return\n"
     "        with torch.no_grad():\n"
     "            xy = self._rigid_body_pos[:, 0, :2]\n"
     "            foot = self._rigid_body_pos[:, :, 2].min(dim=1).values\n"
     "            if not hasattr(self, \"_blk_portals\"):\n"
     "                self._blk_portals = {}\n"
     "                self._blk_portal_pos = {}\n"
     "                for _p in self.mesh_data.get(\"portals\", []):\n"
     "                    _key = (int(_p[\"a\"]), int(_p[\"b\"]))\n"
     "                    self._blk_portals[_key] = torch.tensor(\n"
     "                        _p[\"normal\"], device=self.device,\n"
     "                        dtype=torch.float32)\n"
     "                    self._blk_portal_pos[_key] = torch.tensor(\n"
     "                        _p[\"p\"][:2], device=self.device,\n"
     "                        dtype=torch.float32)\n"
     "            probe = torch.cat([xy, torch.zeros_like(xy[:, :1])],\n"
     "                              dim=-1).unsqueeze(1)\n"
     "            px, py = self.terrain.world_points_to_map(probe.clone())\n"
     "            surf = self.terrain.heightsamples[:, px, py].transpose(0, 1)\n"
     "            cand = ((surf - foot[:, None]).abs() <= 0.15) & (surf > -1.4)\n"
     "            vel = self._rigid_body_vel[:, 0, :2]\n"
     "            cur = self.terrain.block_id.clone()\n"
     "            new = cur.clone()\n"
     "            forced = torch.zeros_like(cur)\n"
     "            for e in range(surf.shape[0]):\n"
     "                for _k, _nrm in self._blk_portals.items():\n"
     "                    if int(cur[e]) != _k[0]:\n"
     "                        continue\n"
     "                    _pp = self._blk_portal_pos[_k]\n"
     "                    if float(torch.norm(xy[e] - _pp)) < 0.45 and \\\n"
     "                            float(vel[e] @ _nrm) > 0.2:\n"
     "                        new[e] = _k[1]\n"
     "                        forced[e] = 1\n"
     "                        break\n"
     "                if int(forced[e]):\n"
     "                    continue\n"
     "                idx = torch.nonzero(cand[e]).flatten().tolist()\n"
     "                if len(idx) == 1:\n"
     "                    new[e] = idx[0]\n"
     "                elif len(idx) > 1 and int(cur[e]) in idx:\n"
     "                    best, best_s = int(cur[e]), -1e9\n"
     "                    for m in idx:\n"
     "                        if m == int(cur[e]):\n"
     "                            continue\n"
     "                        nrm = self._blk_portals.get((int(cur[e]), m))\n"
     "                        if nrm is None:\n"
     "                            rev = self._blk_portals.get((m, int(cur[e])))\n"
     "                            nrm = None if rev is None else -rev\n"
     "                        if nrm is None:\n"
     "                            continue\n"
     "                        s = float((vel[e] * nrm).sum())\n"
     "                        if s > best_s:\n"
     "                            best, best_s = m, s\n"
     "                    if best_s >= 0.3:\n"
     "                        new[e] = best\n"
     "            if not hasattr(self, \"_blk_pending\"):\n"
     "                self._blk_pending = cur.clone()\n"
     "                self._blk_stable = torch.zeros_like(cur)\n"
     "            same = new == self._blk_pending\n"
     "            self._blk_stable = torch.where(same, self._blk_stable + 1,\n"
     "                                           torch.ones_like(self._blk_stable))\n"
     "            self._blk_pending = new.clone()\n"
     "            commit = (self._blk_stable >= 3) & (new != cur)\n"
     "            if bool(commit.any()):\n"
     "                self.terrain.set_block_id(torch.where(commit, new, cur))\n"
     "                self._block_id = self.terrain.block_id\n"
     "\n"
     "    def _compute_observations(self, env_ids=None):\n"
     "        # base_task has no common_step_counter, so re-running this once per\n"
     "        # obs call is the only reliable hook; it is a handful of gathers.\n"
     "        self._update_block_id()\n"
     "        return super()._compute_observations(env_ids)\n"
     "\n"
     "    def create_mesh_ground(self):"),

    # ---- pin the spawn heading to the route direction -----------------------
    # stair_heading_mode == "straight" cancels the clip heading, i.e. pins
    # body +X to +x.  A route that starts along +y would then need a 90 deg
    # turn in place, which the controller never learned, and it just stalls.
    # Scenes may carry an absolute target heading in their pkl.
    ("                ang = torch.atan2(ex[:, 1], ex[:, 0])\n"
     "                half = -ang / 2.0",
     "                ang = torch.atan2(ex[:, 1], ex[:, 0])\n"
     "                half = (float(getattr(self, \"_scene_start_yaw\", 0.0)) - ang) / 2.0"),

    # ---- re-anchor the trajectory when the block changes --------------------
    # One long concatenated polyline makes the controller cross block borders
    # on a stale plan; every block lives in the same world frame, so switching
    # to that block own sub-path needs no coordinate transform at all.
    ("            if bool(commit.any()):\n"
     "                self.terrain.set_block_id(torch.where(commit, new, cur))\n"
     "                self._block_id = self.terrain.block_id",
     "            if bool(commit.any()):\n"
     "                self.terrain.set_block_id(torch.where(commit, new, cur))\n"
     "                self._block_id = self.terrain.block_id\n"
     "                self._apply_block_path(torch.nonzero(commit).flatten())"),

    ("            _sb = 0\n"
     "        self.terrain.set_block_id(torch.full((self.num_envs,), _sb,",
     "            _sb = 0\n"
     "        self._start_block = _sb\n"
     "        self._block_paths = self.mesh_data.get(\"block_paths\")\n"
     "        self.terrain.set_block_id(torch.full((self.num_envs,), _sb,"),

    ("    def create_mesh_ground(self):",
     "    def _apply_block_path(self, env_ids=None):\n"
     "        \"\"\"Write the current block sub-path into the trajectory verts.\n"
     "\n"
     "        All blocks share one world frame, so this is a plain overwrite:\n"
     "        no translation or rotation is needed when the block switches.\n"
     "        \"\"\"\n"
     "        paths = getattr(self, \"_block_paths\", None)\n"
     "        if not paths or not hasattr(self, \"_traj_gen\"):\n"
     "            return\n"
     "        if env_ids is None:\n"
     "            env_ids = torch.arange(self.num_envs, device=self.device)\n"
     "        verts = self._traj_gen._verts\n"
     "        n = verts.shape[1]\n"
     "        for e in env_ids.tolist():\n"
     "            blk = int(self.terrain.block_id[e])\n"
     "            if blk < 0 or blk >= len(paths):\n"
     "                continue\n"
     "            pts = np.asarray(paths[blk], dtype=np.float64)\n"
     "            if len(pts) < 2:\n"
     "                continue\n"
     "            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)\n"
     "            cum = np.concatenate([[0.0], np.cumsum(seg)])\n"
     "            if cum[-1] <= 1e-6:\n"
     "                continue\n"
     "            tgt = np.linspace(0.0, cum[-1], n)\n"
     "            rs = np.stack([np.interp(tgt, cum, pts[:, a])\n"
     "                           for a in range(3)], axis=-1)\n"
     "            verts[e] = torch.as_tensor(rs, dtype=verts.dtype,\n"
     "                                       device=verts.device)\n"
     "\n"
     "    def create_mesh_ground(self):"),

    # ---- reset must start from the scene start block ------------------------
    # A reset taken while another block is active would otherwise place the
    # character at that block height (respawning on the 3F start at 2F height
    # looks exactly like falling through the floor).
    ("        root_states = torch.cat([root_pos, root_rot], dim=1)\n"
     "\n"
     "        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1)",
     "        if getattr(self, \"real_mesh\", False) and hasattr(self, \"_start_block\"):\n"
     "            self.terrain.set_block_id(torch.full((self.num_envs,),\n"
     "                                                 int(self._start_block),\n"
     "                                                 dtype=torch.long,\n"
     "                                                 device=self.device))\n"
     "            self._block_id = self.terrain.block_id\n"
     "        root_states = torch.cat([root_pos, root_rot], dim=1)\n"
     "\n"
     "        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1)"),
]

# (old, new) for base_task.py: record the per-frame block id and dump it
BASE_EDITS = [
    ("            try:\n"
     "                self.state_record[\"obs\"].append(self.obs_buf.detach().cpu().clone())\n"
     "            except Exception:\n"
     "                pass",
     "            try:\n"
     "                self.state_record[\"obs\"].append(self.obs_buf.detach().cpu().clone())\n"
     "            except Exception:\n"
     "                pass\n"
     "            _bid = getattr(self, \"_block_id\", None)\n"
     "            if _bid is not None:\n"
     "                self.state_record.setdefault(\"block_id\", []).append(\n"
     "                    _bid.detach().cpu().clone())"),

    ("        if self.state_record.get(\"obs\"):\n"
     "            payload[\"obs\"] = torch.stack(self.state_record[\"obs\"], dim=0).cpu()",
     "        if self.state_record.get(\"obs\"):\n"
     "            payload[\"obs\"] = torch.stack(self.state_record[\"obs\"], dim=0).cpu()\n"
     "        if self.state_record.get(\"block_id\"):\n"
     "            payload[\"block_id\"] = torch.stack(\n"
     "                self.state_record[\"block_id\"], dim=0).cpu()"),
]


def detect_nl(src):
    return "\r\n" if "\r\n" in src else "\n"


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _apply_file(path, backup, edits, name, first_only, marker):
    src = read(path)
    nl = detect_nl(src)
    if marker in src:
        print("%s: already applied" % name)
        return True
    if not os.path.exists(backup):
        write(backup, src)
        print("%s: backup -> %s" % (name, os.path.basename(backup)))
    for i, (old, new) in enumerate(edits):
        o, n = old.replace("\n", nl), new.replace("\n", nl)
        cnt = src.count(o)
        if i in first_only:
            if cnt < 1:
                print("%s EDIT %d: anchor missing\n  %r" % (name, i, old[:80]))
                return False
            src = src.replace(o, n, 1)
            continue
        if cnt != 1:
            print("%s EDIT %d: anchor count = %d (expected 1)\n  %r"
                  % (name, i, cnt, old[:80]))
            return False
        src = src.replace(o, n)
    write(path, src)
    print("%s: applied %d edits" % (name, len(edits)))
    return True


def apply(root):
    ok = _apply_file(os.path.join(root, T_TARGET), os.path.join(root, T_BACKUP),
                     EDITS, "terrain", FIRST_ONLY, "self.num_blocks = int(")
    ok = ok and _apply_file(os.path.join(root, B_TARGET), os.path.join(root, B_BACKUP),
                            BASE_EDITS, "base_task", set(), "state_record[\"block_id\"]")
    return 0 if ok else 1


def revert(root):
    for path, backup in ((T_TARGET, T_BACKUP), (B_TARGET, B_BACKUP)):
        p, b = os.path.join(root, path), os.path.join(root, backup)
        if os.path.exists(b):
            write(p, read(b))
            os.remove(b)
            print("reverted", path)
    return 0


def status(root):
    src = read(os.path.join(root, T_TARGET))
    base = read(os.path.join(root, B_TARGET))
    marks = [("num_blocks", "self.num_blocks = int(" in src),
             ("block_id tensor", "def set_block_id" in src),
             ("gather in sample_height_points", "heightsamples[_blk, px, py]" in src),
             ("gather in _sample_static_terrain", "self.terrain.heightsamples[_blk, px, py]" in src),
             ("start_block seed", "self._block_id = self.terrain.block_id" in src),
             ("spawn snap", "_new_root = center_height - _delta" in src),
             ("block state machine", "def _update_block_id" in src),
             ("obs hook", "def _compute_observations" in src),
             ("dump record", "state_record[\"block_id\"]" in base),
             ("dump payload", "payload[\"block_id\"]" in base)]
    for name, ok in marks:
        print("  %-34s %s" % (name, "+" if ok else "-"))
    print("  backup present:", os.path.exists(os.path.join(root, T_BACKUP)),
          os.path.exists(os.path.join(root, B_BACKUP)))
    return 0 if all(v for _, v in marks) else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["apply", "revert", "status"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    return {"apply": apply, "revert": revert, "status": status}[a.action](a.root)


if __name__ == "__main__":
    sys.exit(main())
