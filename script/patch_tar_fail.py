"""把 tar_fail（轨迹偏离判定）关掉，只保留 has_fallen（摔倒判定）。

动机：
    路点序列本应只是【引导】。但 tar_fail 把"偏离轨迹 >4 m"也判成失败，
    而"超出路点多少"与"能不能稳定下楼"没有因果关系。
    实测：球间距 0.76 m 时稳态超前 1.41 m（安全）；
          球间距 1.52 m 时稳态超前 3.18 m、峰值 3.72 m（贴着 4 m 阈值）。
    于是 episode 被一个与目标无关的约束杀死。

注意：compute_humanoid_reset 被 @torch.jit.script 装饰，
      TorchScript 不支持 os.environ / __import__，
      所以这里用「应用补丁 = 关掉；--revert = 恢复」的方式做开关。

用法：
    python script/patch_tar_fail.py --apply | --revert | --status
"""

from __future__ import annotations

import argparse
import os
import shutil

TARGET = "pacer/env/tasks/humanoid_pedestrain_terrain.py"
BAK = TARGET + ".orig"

OLD = """        tar_fail = tar_dist_sq > fail_dist * fail_dist"""
NEW = """        # [patched] tar_fail disabled by script/patch_tar_fail.py
        # 超出引导路点不算失败；只有摔倒才算
        tar_fail = torch.zeros_like(reset_buf)"""

MARK = "[patched] tar_fail disabled"


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
        print("tar_fail disabled: %s" % ("YES" if patched else "no"))
        print("backup exists    : %s" % ("yes" if os.path.exists(BAK) else "no"))
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
        print("applied: tar_fail disabled")
        return 0

    if a.revert:
        if not os.path.exists(BAK):
            print("no backup")
            return 1
        shutil.copy2(BAK, TARGET)
        print("reverted: tar_fail restored")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
