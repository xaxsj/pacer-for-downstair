# 根因记录：角色"站不住/一直塌"——`has_upright_start: False` 造成观测与物理解耦

> 日期：2026-11-02
> 适用：`pacer-main` + `stairs05_neg`（5 级楼梯）+ `output/release/pacer/Humanoid.pth`
> 结论一句话：**`pacer/data/cfg/pacer.yaml` 里 `has_upright_start` 必须是 `True`。用 `False` 时物理中的身体是横躺的，但观测把它补偿成直立，于是策略在"以为站着"的情况下输出站姿，实际一直在摔倒。**

---

## 一、症状（当时看到的现象）

| 现象 | 说明 |
|---|---|
| 每 13 帧（0.43 s）必重置一次 | 200 帧里稳定 15 次，周期完全固定 |
| dump 反算出的姿态完全正常 | 用 `root_rot + dof_pos` 反算，身体直立、朝向 177.7°（轨迹 −180°） |
| 但 `--auto_record` 相机里身体是横的 | 水平悬空、四肢张开、往下坠 |
| 换路径长度、换速度、改 `--start_yaw`、关摔倒判定 | **轨迹逐帧逐位相同**，所有参数都不敏感 |
| 零动作测试（`PACER_ZERO_ACTION=1`） | 与有策略**完全相同** |

最后一条是最强的信号：**策略输出对仿真没有任何影响**，说明控制回路在某个环节已经断开。

---

## 二、根因

PACER 的初始位姿由 `_get_fixed_smpl_state_from_motionlib()` 给出，`has_upright_start` 决定是否按 **SMPL base rotation** 摆放：

```
(0.5, 0.5, 0.5, 0.5) = 绕 (1,1,1) 转 120°，把 body +z 映射到 +x
```

- `has_upright_start: False`
  - **物理**：初始 root_rot 带着 base rotation → 身体在世界坐标系里**横躺**（图二）
  - **观测**：`remove_base_rot()` 把四元数"修正"回直立再喂给策略
  - 结果：策略看到一个站着的自己，实际是个躺着的身体 → 任何动作都等于在地上挣扎
- `has_upright_start: True`
  - 物理与观测都不做补偿，两者一致 → 真正站立

### 为什么排查时一直被误导

`--auto_record` 导出的 dump 存的是**观测侧张量**：

```python
# pacer/env/tasks/base_task.py : _dump_auto_record()
payload["root_states"]    = self._humanoid_root_states   # 观测侧
payload["rigid_body_pos"] = self._rigid_body_pos         # 观测侧（还被 "ZL hack" 覆盖过）
```

而视频是 Isaac Gym 渲染的**仿真实况**。所以：

| 证据 | 来源 | 受害于 bug 吗 |
|---|---|---|
| dump / `render_path.py` 重放 | 观测侧 root + dof 反算 | ✅ 看着完全正常 |
| `--auto_record` 视频 | 仿真渲染 | ❌ 暴露真相：横躺 |

于是出现"数据一切正常，画面一直在摔"的诡异局面。

---

## 三、修复与验证

### 修复

```yaml
# pacer/data/cfg/pacer.yaml
has_upright_start: True
```

### 效果（`stairs05_neg`，5 级，`--start_yaw 1.5708`，200 帧）

| 指标 | `False`（bug） | `True`（修复后） |
|---|---|---|
| 重置次数 | 15 | **1** |
| dz | −0.88（每 13 帧塌一次） | **−0.814（走完整段 0.75 m 楼梯）** |
| dx | −0.18 | **−1.303** |
| 末态 root z | 0.852（趴着） | **0.922（站着到底）** |

对应的历史记录也被解释清楚了：本文档库里那条"5 级 + 路径 −x 走下一段（x→−1.34，z 1.74→0.84）"**本来就是 `True` 下跑出来的**，后来"因为重置更多所以回滚成 False"才是把一切搞坏的那一步。

---

## 四、可复用的判据（重要）

### 铁律：**判断角色姿态，只能看仿真相机，不能看 dump 反算**

```bash
# ✅ 真相源：仿真渲染（auto_record 相机）
bash script/run_stairs.sh --mesh stairs05_neg --yaw 1.5708 --frames 200 --tag check
# → experiment_result/<tag>/auto.mp4

# ❌ 会骗人的：dof/root 反算重放
python script/render_path.py --states ... --out replay.mp4
```

`render_path.py` 仍然有用（画轨迹、看朝向），但**它只能验证"root+dof 记录了什么"，不能验证仿真里发生了什么**。两者在 `has_upright_start` 错误时是矛盾的。

### 用"参数不敏感"作为解耦信号

如果改动下列任一参数后轨迹**逐帧几乎不变**，说明控制回路已断，不要再调参数了，先查状态一致性：

- 路径长度 / 隐含速度
- `--start_yaw`
- 摔倒判定开关
- 策略输入（零动作 vs 正常）

正常情况下，零动作和正常策略的轨迹**必须明显不同**。

---

## 五、排查工具（`script/` 下，均已验证可用）

| 脚本 | 作用 |
|---|---|
| `run_stairs.sh` | 实验启动器；产物落 `experiment_result/<tag>/`（states + auto.mp4 + run.log）；`--flat` 跑平地对照 |
| `diag_states.py` | 轨迹总览：位移、漂移、重置次数/周期、逐帧表 |
| `diag_dof.py` | 关节运动分析：哪些 dof 在动、幅度、是否交替 |
| `check_heading.py` | **按本仓库口径**同时输出 body +X 与视觉正面（脚趾−脚踝） |
| `inspect_mesh.py` | 地形几何：平台边缘、台阶剖面、高度图量级校验 |
| `check_collision_vs_heightmap.py` | 小球跌落测碰撞面，与高度图逐点比对 |
| `drop_test.py` | 人偶静置跌落，分离"碰撞"与"控制" |
| `pose_probe.py` | 按 root+dof 反算真实刚体世界坐标 |
| `compare_sim_obs.py` | 比对 dump 的观测侧 root 与仿真侧 root |
| `patch_rb_hack.py` / `patch_zero_action.py` / `patch_record_sim.py` | 诊断开关（默认关，可 `--revert`） |

---

## 六、本轮同时被证伪的其他假设

留档以免重复踩：

| 假设 | 检验 | 结论 |
|---|---|---|
| 路径隐含速度太慢（总结旧主线） | 2.4 m vs 8.6 m 路径对比 | ❌ 逐帧轨迹逐位相同 |
| 出生点贴着台阶边缘 | `inspect_mesh.py` | ❌ 平台 x∈[−0.8,0.4]，还有 0.8 m 跑道 |
| 高度图量纲错（`x_scale=20`） | 代入 `world_points_to_map` 公式 | ❌ 与网格 bbox 完全吻合 |
| 脚下没有碰撞面 | `check_collision_vs_heightmap.py` | ❌ 碰撞面 = 高度图 +2 mm |
| 地形网格有位移 | 读 `create_mesh_ground()` | ❌ `transform.p=(0,0,0)` |
| 观测被注入了 motion-lib 的假刚体（ZL hack） | A/B：`PACER_NO_RB_HACK` | ❌ 两边完全一致 |
| 高度观测是绝对值（平台 0.75 超出训练分布） | 读 obs 代码 389/393 行 | ❌ 用的是 `center_height − measured`，相对量 |
| 出生朝向错 | 指南 SOP 标定 + 方法 A 渲染 | ❌ 视觉正面 177.7°，轨迹 −180°，残差 2.3° |

---

## 七、仍待解决

**横向漂移**：下楼成功，但侧向偏了 0.78 m（最大 1.13 m），楼梯只有 1.2 m 宽，基本擦边下去。这是本次之后唯一剩下的主要问题。
