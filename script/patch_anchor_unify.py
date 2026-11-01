"""把「路点追踪」整体改成角色锚定：观测 / 终止 / 奖励三处共用一个函数，并默认开启。

背景：
    原始实现里三处都用全局时间轴：
        timestep_beg = progress_buf * dt                     (观测)
        tar_pos = calc_pos(env_ids, time)                    (终止 tar_fail)
        tar_pos = calc_pos(env_ids, time)                    (奖励)
    角色步速一旦与参考速度有微小差异就会累积超前（实测稳态 1.41 m），
    路点落到身后，策略转身倒退并摔倒。

改法：
    新增 _anchored_time(env_ids, sample_offset)：
        把角色 XY 投影到轨迹折线上得到相位 u，换算成等效时间。
    三处统一改为调用它：
        观测  _fetch_traj_samples   offset=0   -> 路点固定在角色正前方
        终止  _compute_reset        offset=0   -> tar_fail 只反映横向偏离
        奖励  _compute_reward       offset=1   -> 奖励瞄准「下一个路点」，避免退化成常数

    默认值改为 char（PACER_TRAJ_ANCHOR=time 可回到旧行为）。

用法：
    python script/patch_anchor_unify.py --apply | --revert | --status
"""

from __future__ import annotations

import argparse
import os
import shutil

TRAJ = "pacer/env/tasks/humanoid_traj.py"
TERR = "pacer/env/tasks/humanoid_pedestrain_terrain.py"

HELPER = '''    def _anchored_time(self, env_ids, sample_offset=0):
        """角色锚定的等效时间。

        PACER_TRAJ_ANCHOR=char（默认）：把角色 XY 投影到轨迹折线上得到相位 u，
        再换算成等效时间；sample_offset 以 traj_sample_timestep 为单位向前偏移。
        于是观测 / 终止 / 奖励都以「角色当前位置」为参考，而不是全局时间轴。
        设 PACER_TRAJ_ANCHOR=time 可回到原来的纯时间推进。

        PACER_VIEW_MIN（默认 1.0 m）：视野下限，即最近一个路点与角色的距离。
        整组路点整体前移（相邻间距不变）。用于把「脚下 / 身后」的区域整段让出来，
        避免策略把离自己太近的点当成目标而转向（backpedal）。
        二维扫描（dt x view_min）实测 1.0 m 最优：
        dt=2.0/view_min=1.0 的航向误差 5.0 度，而 view_min=0 为 14.4 度、
        view_min=2.0 为 19.7 度（2 m 盲区会把策略逼成冲刺）。
        """
        if __import__("os").environ.get("PACER_TRAJ_ANCHOR", "char") != "char":
            return self.progress_buf[env_ids] * self.dt
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
        _total = _cum[:, -1].clamp_min(1e-9)
        _lead = float(__import__("os").environ.get("PACER_VIEW_MIN", "1.0"))
        _dur = self._traj_gen.get_traj_duration()
        _u = (_arc / _total).clamp(0.0, 1.0)
        return (_u * _dur + (_lead / _total) * _dur
                + float(sample_offset) * self._traj_sample_timestep)

'''

EDITS = [
    # 1) 插入统一辅助函数
    (TRAJ,
     "    def _fetch_traj_samples(self, env_ids=None):",
     HELPER + "    def _fetch_traj_samples(self, env_ids=None):"),
    # 2) 观测：改用统一函数
    (TRAJ,
     "        timestep_beg = self.progress_buf[env_ids] * self.dt",
     "        timestep_beg = self._anchored_time(env_ids)"),
    # 3) 终止：tar_fail 只反映横向偏离
    (TERR,
     "        tar_pos = self._traj_gen.calc_pos(env_ids, time)\n        ### ZL: entry point",
     "        tar_pos = self._traj_gen.calc_pos(env_ids, self._anchored_time(env_ids, 0))\n        ### ZL: entry point"),
    # 4) 奖励：瞄准下一个路点
    (TERR,
     "        tar_pos = self._traj_gen.calc_pos(env_ids, time)\n        if self.fuzzy_target:",
     "        tar_pos = self._traj_gen.calc_pos(env_ids, self._anchored_time(env_ids, 1))\n        if self.fuzzy_target:"),
]

MARK = "_anchored_time"


def _read_normalized(path):
    """Read as UTF-8, normalise CRLF -> LF, and report whether the file used CRLF.

    Matching on LF-only text keeps the OLD/NEW literals simple; the original
    line endings are restored on write so the diff stays minimal.
    """
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def _write_restored(path, text, crlf):
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    for f in (TRAJ, TERR):
        if not os.path.exists(f):
            print("not found: %s (run from repo root)" % f)
            return 1

    traj_src, traj_crlf = _read_normalized(TRAJ)
    terr_src, terr_crlf = _read_normalized(TERR)
    crlf = {TRAJ: traj_crlf, TERR: terr_crlf}
    done = "def _anchored_time" in traj_src

    if a.status or not (a.apply or a.revert):
        print("applied: %s" % ("YES" if done else "no"))
        for f in (TRAJ, TERR):
            print("  backup %-50s %s" % (f + ".orig", "yes" if os.path.exists(f + ".orig") else "no"))
        print("  remaining raw tar_pos(time): %d" % terr_src.count("calc_pos(env_ids, time)"))
        print("  line endings: traj=%s terr=%s" % (
            "CRLF" if traj_crlf else "LF", "CRLF" if terr_crlf else "LF"))
        return 0

    if a.apply:
        if done:
            print("already applied")
            return 0
        srcs = {TRAJ: traj_src, TERR: terr_src}
        for f, old, new in EDITS:
            n = srcs[f].count(old)
            if n != 1:
                print("match failed in %s (n=%d) for: %s" % (f, n, old.splitlines()[0][:60]))
                return 1
            srcs[f] = srcs[f].replace(old, new)
        for f in (TRAJ, TERR):
            if not os.path.exists(f + ".orig"):
                shutil.copy2(f, f + ".orig")
                print("backup -> %s.orig" % f)
            _write_restored(f, srcs[f], crlf[f])
        print("applied: unified character-anchored time (default=char)")
        return 0

    if a.revert:
        ok = True
        for f in (TRAJ, TERR):
            if os.path.exists(f + ".orig"):
                shutil.copy2(f + ".orig", f)
                print("reverted %s" % f)
            else:
                ok = False
                print("no backup for %s" % f)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
