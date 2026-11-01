"""在 --auto_record 的 dump 里额外记录**仿真侧** root 状态，用于和观测侧比对。

背景：_dump_auto_record() 存的是 self._humanoid_root_states（观测侧），
      而 Isaac Gym 仿真的真实 root 在 self._root_states（humanoid.py:157）。
      若两者不一致，说明物理与策略看到的状态被解耦了 ——
      这正是"反算出来的姿态是对的、画面里却是横的"的成因。

用法：
    python script/patch_record_sim.py --apply | --revert | --status
之后 dump 会多出 sim_root_states (T, E, 13) 字段。
"""

from __future__ import annotations

import argparse
import os
import shutil

TARGET = "pacer/env/tasks/base_task.py"
BAK = TARGET + ".orig"

OLD1 = """            if hasattr(self, "_rigid_body_pos"):
                self.state_record["rigid_body_pos"].append(
                    self._rigid_body_pos.cpu().clone())"""

NEW1 = """            if hasattr(self, "_rigid_body_pos"):
                self.state_record["rigid_body_pos"].append(
                    self._rigid_body_pos.cpu().clone())
            # [patched] 同时记录仿真侧 root（self._root_states），用于比对观测侧
            if "_sim_root" not in self.state_record:
                self.state_record["_sim_root"] = []
            self.state_record["_sim_root"].append(
                self._root_states.clone().cpu())"""

OLD2 = """        if self.state_record.get("rigid_body_pos"):
            payload["rigid_body_pos"] = torch.stack(
                self.state_record["rigid_body_pos"], dim=0).cpu()"""

NEW2 = """        if self.state_record.get("rigid_body_pos"):
            payload["rigid_body_pos"] = torch.stack(
                self.state_record["rigid_body_pos"], dim=0).cpu()
        # [patched] 仿真侧 root
        if self.state_record.get("_sim_root"):
            payload["sim_root_states"] = torch.stack(
                self.state_record["_sim_root"], dim=0).cpu()"""


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
    patched = "[patched] 同时记录仿真侧 root" in src

    if a.status or not (a.apply or a.revert):
        print("已打补丁: %s" % ("是" if patched else "否"))
        print("备份存在: %s" % ("是" if os.path.exists(BAK) else "否"))
        return 0

    if a.apply:
        if patched:
            print("已打过补丁，跳过")
            return 0
        if src.count(OLD1) != 1 or src.count(OLD2) != 1:
            print("匹配失败 (%d, %d)" % (src.count(OLD1), src.count(OLD2)))
            return 1
        if not os.path.exists(BAK):
            shutil.copy2(TARGET, BAK)
            print("已备份 -> %s" % BAK)
        open(TARGET, "w", encoding="utf-8").write(
            src.replace(OLD1, NEW1).replace(OLD2, NEW2))
        print("补丁已应用")
        return 0

    if a.revert:
        if not os.path.exists(BAK):
            print("没有备份")
            return 1
        shutil.copy2(BAK, TARGET)
        print("已回滚")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
