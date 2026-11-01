"""给 PD 目标计算加一个「零动作」开关，用于初始保持测试。

背景：pacer/env/tasks/humanoid.py:1278 的 _action_to_pd_targets()
      pd_tar = pd_action_offset + pd_action_scale * action
      策略输出 action 决定目标关节角。

本脚本加开关（默认关，行为完全不变）：
    PACER_ZERO_ACTION=1 -> action 不起作用，目标恒等于默认站姿 _pd_action_offset

判据：
    零动作能站住  -> PD/驱动正常，塌下是策略输出的姿势造成的
    零动作仍然塌  -> PD 增益/目标映射有问题，与策略无关

用法：
    python script/patch_zero_action.py --apply | --revert | --status
"""

from __future__ import annotations

import argparse
import os
import shutil

TARGET = "pacer/env/tasks/humanoid.py"
BAK = TARGET + ".orig"

OLD = """    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar"""

NEW = """    def _action_to_pd_targets(self, action):
        # [patched] PACER_ZERO_ACTION=1 -> 零动作，目标恒为默认站姿
        if os.environ.get("PACER_ZERO_ACTION", "0") == "1":
            # offset 可能是 (D,) 或 (1,D)，用广播复制到 action 的形状
            return (self._pd_action_offset + 0.0 * action).clone()
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar"""


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
    patched = "[patched] PACER_ZERO_ACTION" in src

    if a.status or not (a.apply or a.revert):
        print("已打补丁: %s" % ("是" if patched else "否"))
        print("备份存在: %s" % ("是" if os.path.exists(BAK) else "否"))
        print("文件里有 import os: %s" % ("是" if "\nimport os" in src or src.startswith("import os") else "否"))
        return 0

    if a.apply:
        if patched:
            print("已打过补丁，跳过")
            return 0
        if "\nimport os" not in src and not src.startswith("import os"):
            print("警告：文件顶部没有 import os，补丁会失败于运行时")
        if src.count(OLD) != 1:
            print("匹配失败 (n=%d)，未修改" % src.count(OLD))
            return 1
        if not os.path.exists(BAK):
            shutil.copy2(TARGET, BAK)
            print("已备份 -> %s" % BAK)
        open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW))
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
