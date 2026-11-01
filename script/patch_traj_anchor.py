"""把路点采样起点从「时间进度」改成「角色在路径上的投影相位」。

动机：
    现状 timestep_beg = progress_buf * dt  -> 路点按时间往前走，角色跑快就会把
    路点甩到身后，观测里出现"目标在背后"，策略于是转身/倒退，最后摔倒。

    改成锚定角色后：10 个球永远固定在角色【正前方】 k*traj_sample_timestep 处，
    "超前"在定义上消失。观测语义完全不变（仍是 10 个相对位置），不需要重训。

开关：
    PACER_TRAJ_ANCHOR=char  -> 锚定角色（新行为）
    未设置 / =time          -> 与原来完全一致

用法：
    python script/patch_traj_anchor.py --apply | --revert | --status
"""

from __future__ import annotations

import argparse
import os
import shutil

TARGET = "pacer/env/tasks/humanoid_traj.py"
BAK = TARGET + ".orig"

OLD = """        timestep_beg = self.progress_buf[env_ids] * self.dt"""

NEW = '''        # [patched] PACER_TRAJ_ANCHOR=char -> 路点锚定在角色身上（局部实时目标）
        if __import__("os").environ.get("PACER_TRAJ_ANCHOR", "time") == "char":
            _v = self._traj_gen._verts[env_ids]
            _root_xy = self._humanoid_root_states[env_ids, 0:2]
            _a = _v[:, :-1, :2]
            _ab = _v[:, 1:, :2] - _a
            _L2 = (_ab * _ab).sum(-1, keepdim=True).clamp_min(1e-9)
            _t = (((_root_xy.unsqueeze(1) - _a) * _ab).sum(-1, keepdim=True) / _L2).clamp(0.0, 1.0)
            _p = _a + _t * _ab
            _d = ((_root_xy.unsqueeze(1) - _p) ** 2).sum(-1)
            _i = _d.argmin(dim=-1)
            _seg = _ab.norm(dim=-1)
            _cum = torch.cat(
                [torch.zeros(_seg.shape[0], 1, device=_seg.device, dtype=_seg.dtype),
                 _seg.cumsum(-1)], dim=-1)
            _arc = (_cum.gather(1, _i.unsqueeze(-1)).squeeze(-1)
                    + _t.squeeze(-1).gather(1, _i.unsqueeze(-1)).squeeze(-1)
                    * _seg.gather(1, _i.unsqueeze(-1)).squeeze(-1))
            _u = (_arc / _cum[:, -1].clamp_min(1e-9)).clamp(0.0, 1.0)
            timestep_beg = _u * self._traj_gen.get_traj_duration()
        else:
            timestep_beg = self.progress_buf[env_ids] * self.dt'''

MARK = "[patched] PACER_TRAJ_ANCHOR"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(TARGET):
        print("not found: %s (run from repo root)" % TARGET)
        return 1
    src = open(TARGET, encoding="utf-8").read()
    patched = MARK in src

    if a.status or not (a.apply or a.revert):
        print("patched: %s" % ("YES" if patched else "no"))
        print("backup : %s" % ("yes" if os.path.exists(BAK) else "no"))
        n = src.count(OLD)
        print("target line occurrences: %d %s" % (n, "(ok)" if n == 1 else "(WARNING)"))
        return 0

    if a.apply:
        if patched:
            print("already applied")
            return 0
        if src.count(OLD) != 1:
            print("match failed (n=%d)" % src.count(OLD))
            return 1
        if not os.path.exists(BAK):
            shutil.copy2(TARGET, BAK)
            print("backup -> %s" % BAK)
        open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW))
        print("applied: PACER_TRAJ_ANCHOR=char enables character-anchored waypoints")
        return 0

    if a.revert:
        if not os.path.exists(BAK):
            print("no backup")
            return 1
        shutil.copy2(BAK, TARGET)
        print("reverted")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
