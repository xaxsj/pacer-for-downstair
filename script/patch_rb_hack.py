"""给 PACER 的 ZL rb-pos hack 加一个环境变量开关，用于 A/B 对照。

背景：pacer/env/tasks/humanoid_amp.py 的 _set_env_state() 会把 motion lib 里的
rigid_body_pos 写进**观测张量** self._rigid_body_pos，然后在 _refresh_sim_tensors()
里用 "_reset_rb_pos" 把它盖回刷新后的仿真值（代码注释写着 "ZL: Hack"）。

问题：motion lib 的刚体坐标与本 checkpoint 的观测坐标系未必一致
（has_upright_start=False 时存在 SMPL base rotation），
这会让策略在重置那一帧看到"姿势错误的自己"。

本脚本只加开关，不改逻辑：
    PACER_NO_RB_HACK=1  -> 跳过注入，观测直接用仿真刷新值
    PACER_NO_RB_HACK 未设 -> 与原来完全一致

用法：
    python script/patch_rb_hack.py --apply
    python script/patch_rb_hack.py --revert
    python script/patch_rb_hack.py --status
"""

from __future__ import annotations

import argparse
import os
import shutil

TARGET = "pacer/env/tasks/humanoid_amp.py"
BAK = TARGET + ".orig"

OLD1 = """            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()"""

NEW1 = """            # [patched] PACER_NO_RB_HACK=1 时不保存，也就不会在下方盖回观测
            if os.environ.get("PACER_NO_RB_HACK", "0") != "1":
                self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
                self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()"""

OLD2 = """        if self._state_reset_happened and "_reset_rb_pos" in self.__dict__:"""

NEW2 = """        if (self._state_reset_happened and "_reset_rb_pos" in self.__dict__
                and os.environ.get("PACER_NO_RB_HACK", "0") != "1"):
            # [patched] 见 script/patch_rb_hack.py"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(TARGET):
        print("找不到 %s（请在 repo 根目录执行）" % TARGET)
        return 1
    src = open(TARGET, encoding="utf-8").read()
    patched = "[patched] PACER_NO_RB_HACK" in src

    if a.status or not (a.apply or a.revert):
        print("已打补丁: %s" % ("是" if patched else "否"))
        print("备份存在: %s" % ("是" if os.path.exists(BAK) else "否"))
        return 0

    if a.apply:
        if patched:
            print("已经是打过补丁的状态，跳过")
            return 0
        if not os.path.exists(BAK):
            shutil.copy2(TARGET, BAK)
            print("已备份 -> %s" % BAK)
        n1 = src.count(OLD1)
        n2 = src.count(OLD2)
        if n1 != 1 or n2 != 1:
            print("匹配失败 (n1=%d, n2=%d)，未修改" % (n1, n2))
            return 1
        out = src.replace(OLD1, NEW1).replace(OLD2, NEW2)
        open(TARGET, "w", encoding="utf-8").write(out)
        print("补丁已应用")
        return 0

    if a.revert:
        if not os.path.exists(BAK):
            print("没有备份，无法回滚")
            return 1
        shutil.copy2(BAK, TARGET)
        print("已从 %s 回滚" % BAK)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
