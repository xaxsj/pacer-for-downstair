# Stage 7 / Stage 8 训练计划

> 目标：从"能在结构化场景里下楼"推进到"人群仿真级导航"——真实 3D 建模地面、高层指定路径、楼梯上允许非预期转向。
>
> 制定日期：2026-11-10
> 前置工作：见 `docs/PACER-多层建筑-A星路线-交接总结.docx`

---

## 0. 目标与约束

**最终目标**：人群仿真级导航 —— 真实 3D 建模地面 + 高层指定路径，能在楼梯上做"非预期转向"。

| 约束 | 来源 | 现状 |
|---|---|---|
| 风格 / 任务权重 = **0.3 / 0.7** | 用户指定 | 当前 0.5 / 0.5（`disc_reward_w` / `task_reward_w`）|
| 地面 = 真实 3D 网格（trimesh），路径由高层规划给出 | 用户指定 | 已有 `--path_file` 机制，训练时也走这条路 |
| 楼梯格式同 stage5（riser 0.08–0.22，tread 0.26–0.38，8–12 级）| 用户指定 | stage5 配置已具备 |
| 每阶段 750 轮 | 用户指定 | stage6 为 500 轮 |

---

## 1. 两个 Stage 的定义

### 1.1 Stage 7 —— 结构化路线 + 直楼梯

| 维度 | 设定 |
|---|---|
| 路线 | 每个场景 **2–3 个转弯段** |
| **楼梯上** | **只允许直线**（沿坡向，无转弯）|
| **平地上** | 除转弯处外全部直线 |
| 楼梯 | 上 / 下直楼梯，参数 = stage5 |
| 楼层 | 2–3 层，含上行与下行 |

**路线模板（3 类，训练时随机抽）**

```
T1  下行三弯
    3F ──直楼梯下──> 2F 平台 ─[90°转弯]─> 走廊 ─[90°转弯]─> 楼梯口
       ──直楼梯下──> 1F ──直线──> 终点

T2  单层三弯
    起点 ──直线──> [L 弯] ──直线──> [U 弯 180°] ──直线──> [L 弯]
         ──直线──> 楼梯上/下 ──> 终点

T3  折返
    2F ──直楼梯上──> 3F ─[180° 折返平台]─> ──直楼梯下──> 2F ──直线──> 终点
```

**转弯几何**：圆弧，半径在 **0.9–2.0 m 之间随机**。

> 依据：实测半径 1.5 m 可靠通过、1.0 m 临界、2.0 m 以上会出现"蠕行"。训练必须覆盖整个区间，让策略学会"窄弯减速、宽弯正常走"。

### 1.2 Stage 8 —— Stage 7 + 楼梯降难 + 小幅路径扭曲

| 维度 | 设定 |
|---|---|
| 基础 | **完全继承 Stage 7** 的路线与场景 |
| 楼梯难度 | **降至 stage3**（riser 上界降低、级数减少）|
| **路径扭曲** | 除转弯处外，**上层随机加小幅扭曲，包括楼梯上** |
| 扭曲幅度 | **必须小**：横向 ±0.08–0.15 m，波长 1.5–3.0 m，航向抖动 ±3–5° |
| 目的 | 人群仿真中角色被挤一下就会偏 —— 让策略能容忍"楼梯上莫名转向" |

**扭曲实现**（关键：不能破坏"楼梯只走直线"的可解性）

```python
# 在已生成的路径点上叠一层平滑扰动
noise   = sum_k A_k * sin(2*pi*s/L_k + phi_k)   # s = 弧长
lateral = clip(noise, -0.15, +0.15)              # 单位：米
# 只在"非转弯段"叠加；楼梯段允许叠加，但幅度减半（±0.05）
```

---

## 2. 训练配置改动

新建 `pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_stage7.yaml` 与 `..._stage8.yaml`，从 stage5 的 train 配置复制后修改：

```yaml
params:
  config:
    # ---- 用户要求的权重 ----
    task_reward_w: 0.7        # 原 0.5
    disc_reward_w: 0.3        # 原 0.5

    # ---- 保持不变 ----
    disc_coef: 5
    disc_reward_scale: 2
    disc_logit_reg: 0.01
    disc_grad_penalty: 5
```

对应环境配置 `pacer/data/cfg/pacer_stage7.yaml` / `pacer_stage8.yaml`（从 `pacer_stage5.yaml` 复制）：

```yaml
env:
  # 路线相关（新增开关，见第 3、4 节）
  route_mode: mesh_3d                # 真实 3D 网格 + 高层路径
  route_turn_count: [2, 3]           # 每条路线 2–3 个转弯段
  route_turn_radius: [0.9, 2.0]      # 转弯半径区间（米）
  route_stairs_straight_only: true   # 楼梯上禁止转弯
  route_distort_amp: 0.0             # stage7 = 0；stage8 = 0.08–0.15
  route_distort_on_stairs: 0.0       # stage7 = 0；stage8 = 0.05

  # 楼梯（stage7 = stage5 参数）
  stair_riser_min: 0.08
  stair_riser_max: 0.22
  stair_tread_min: 0.26
  stair_tread_max: 0.38
  stair_up_steps: [8, 12]
  stair_down_steps: [8, 12]

  # stage8 改为 stage3 的楼梯难度
```

---

## 3. 奖励函数：新增"速度惩罚项"

现有奖励已有 `enable_velocity_reward` / `velocity_reward_weight: 0.35`，但那是**奖励速度匹配目标**，并不惩罚"停住"。

实测的失败模式是：**拐角速度从 2.4 m/s 掉到 0.01 m/s，之后起不来**。因此新增一项**停滞惩罚**。

### 3.1 设计

```python
v_xy      = ||root_lin_vel[:, :2]||
v_target  = 目标轨迹在该时刻的隐含速度        # 已有
warm_up   = clip(step / 4000, 0, 1)          # 与现有项共用预热

relu_low  = relu(V_STALL - v_xy) * (v_target > V_STALL)   # 只在"该走却没走"时罚
penalty   = -W_STALL * warm_up * clip(relu_low / V_STALL, 0, 1)

reward += penalty
```

### 3.2 参数

| 符号 | 值 | 说明 |
|---|---|---|
| `V_STALL` | **0.60 m/s** | 低于此速度视为停滞 |
| `W_STALL` | **0.15** | 与 `safe_reward_clip: 0.30` 同量级，不能压过跟踪奖励 |
| 预热 | 4000 步 | 与现有 `safe_reward_warmup_steps` 一致 |
| 门控 | `v_target > V_STALL` | 终点停下时**不罚**（关键，否则学不会"到站停下"）|

### 3.3 新增配置项

```yaml
enable_stall_penalty: true
stall_speed_threshold: 0.60
stall_penalty_weight: 0.15
stall_penalty_warmup_steps: 4000
stall_penalty_only_when_target_moving: true
```

> **注意**：`stall_speed_threshold` 必须低于训练速度下界的一半；PPO horizon 为 32 步，0.6 m/s 阈值下需走 0.64 m 才达标，不会误伤正常减速。

---

## 4. 代码改动清单

| # | 文件 | 改动 | 工作量 |
|---|---|---|---|
| 1 | `pacer/env/tasks/humanoid_pedestrain_terrain.py` | `_compute_reward` 中新增 `_stall_penalty()` | 小 |
| 2 | 同上 | `_reset_ref_state_init`：`route_mode=mesh_3d` 时从场景清单抽一条路线作为 `path_file` | 中 |
| 3 | `pacer/env/util/fixed_path.py` | `apply_to_verts` 支持**相对起点**与**多段圆弧**（目前只吃折线）| 中 |
| 4 | **新增** `script/make_route_library.py` | 生成 stage7/8 路线库：模板 × 随机参数 → 一批 `data/mesh/s7_*.pkl` + `data/paths/s7_*.json` | **大** |
| 5 | **新增** `script/distort_path.py` | 给路径叠加第 1.2 节的小幅平滑扰动 | 小 |
| 6 | `pacer/data/cfg/pacer_stage7.yaml` / `_stage8.yaml` | 见第 2 节 | 小 |
| 7 | `pacer/data/cfg/train/rlg/..._stage7.yaml` / `_stage8.yaml` | 权重 0.3 / 0.7 | 小 |
| 8 | `script/eval_stage7.sh` | 批量评测 + 指标统计（复用现有 eval 脚本）| 小 |

**路线库规模建议**：stage7 **40–60 条**路线（3 类模板 × 随机参数），stage8 在其上再生成 **40 条**带扰动的版本。每条路线配一个 3D 网格（trimesh）+ 一份路径 JSON。

---

## 5. 训练计划

### 5.1 Stage 7：750 轮

| 轮次区间 | 课程内容 | 目的 |
|---|---|---|
| 1 – 100 | 单层平地 + **1 个转弯**，半径 1.5–2.0 m | 稳住基本跟踪 |
| 101 – 250 | 单层 + 2 个转弯，半径 1.2–2.0 m | 学会连续两弯 |
| 251 – 450 | 加直楼梯（上 / 下），楼梯上**严格直线** | 学会楼梯段（沿用 stage5 能力）|
| 451 – 750 | 全模板 T1 / T2 / T3，3 个转弯，半径 **0.9**–2.0 m | 最难档，含窄弯 |

### 5.2 Stage 8：750 轮

| 轮次区间 | 课程内容 |
|---|---|
| 1 – 150 | 继承 stage7 路线，楼梯难度 → stage3，**无扰动**（先适应新楼梯）|
| 151 – 400 | 平地扰动 ±0.08 m，波长 3.0 m |
| 401 – 750 | 平地 ±0.15 m / 1.5 m 波长 + **楼梯上 ±0.05 m** |

### 5.3 训练超参与算力

```
num_envs        : 同 stage6（建议 4096；显存不足降 2048）
horizon         : 32
episodeLength   : 3000 步
保存间隔        : 50 轮
起始权重        : stage7 从 stage6_500ep 热启动；stage8 从 stage7 末轮热启动
```

**算力估算**：stage6 为 500 轮。按同规模，750 轮 × 2 阶段 ≈ **3–5 天单卡**；双卡并行两个 stage 可压到 **2–3 天**。

---

## 6. 验收指标（量化）

每个 stage 训完后，在**固定测试集**上评测（三层场景 mfsb2 + 一条 3 转弯新场景 + 楼梯扰动场景），每项 20 个种子：

| 指标 | Stage 7 目标 | Stage 8 目标 |
|---|---|---|
| 整条路线完成率 | ≥ 80% | ≥ 75% |
| 摔倒次数 / 回合 | ≤ 0.2 | ≤ 0.3 |
| **拐角最低速度** | **≥ 0.5 m/s** | ≥ 0.4 m/s |
| 拐角航向误差 | ≤ 25° | ≤ 30° |
| 最低躯干高度 | ≥ 0.60 | ≥ 0.55 |
| 楼梯段最低速度 | ≥ 0.6 m/s | ≥ 0.5 m/s |
| 平均速度 | 0.8–1.6 m/s | 0.8–1.5 m/s |

**对照基线（当前实测）**：拐角速度 0.01 m/s、62.4% 的帧低于 0.1 m/s、路线只跑完 3F → 2F（约 40%）。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 0.3 / 0.7 权重让动作变得"非人" | 每 100 轮抽 3 条视频人工检查；明显僵硬则把 `disc_reward_w` 回调到 0.4 |
| 停滞惩罚被滥用（学会原地踏步骗奖励）| 门控 `v_target > V_STALL` + 惩罚上限 `clip`；同时盯 `impact_reward` |
| 窄弯（0.9 m）训不出来 | 从 1.5 m 退火到 0.9 m；训不动就停在 1.1 m |
| 路线库与真实场景分布不符 | 用 mfsb2 三层场景作为**保留测试集**，绝不进训练 |
| 楼梯降难（stage3）导致楼梯能力退化 | stage8 保留 20% 样本仍使用 stage5 难度楼梯（混合采样）|

---

## 8. 交付物

1. 4 个配置文件（env × 2 + train × 2）
2. 2 个代码补丁（速度惩罚项、路线库接入）
3. `script/make_route_library.py` + 80–100 条路线库
4. 两个 checkpoint（stage7 / stage8 末轮）
5. 评测报告（第 6 节全部指标）+ 每类模板各 1 段视频
6. 一张对照表：stage5 / stage6 / stage7 / stage8 在同一测试集上的指标

---

## 9. 待确认决策

1. **转弯半径下界**：定 **0.9 m** 还是保守一点 **1.1 m**？
   （0.9 m 接近人挤人时的实际半径，但训练难度大）
2. **路线库规模**：40 条还是 80 条？
   （条数越多泛化越好，但生成时间与磁盘占用上升）
3. **停滞阈值 0.60 m/s** 是否合适？
   （若通过时平均速度约 1.2 m/s，0.6 约为半个身位/秒；想更激进可提到 0.8）

确认后，先做**路线库生成器 + 速度惩罚补丁**（这两块是阻塞项），配置与训练命令随后跟上。

---

## 10. 速度 / 角速度指令改造（2026-11-10 采纳，覆盖 §3 的判定口径）

### 10.1 动机：问题不在“速度奖励权重大小”，而在**指令缺少角速度**

实测（`s3route_r15_t37_w28_0`，Stage 5 权重，side camera rollout）：

| 位置 | 实测速度 | 说明 |
|---|---:|---|
| 直道（t=1.8 s） | **6.25 m/s** | 巡航 |
| 90° 拐角（t=2.2 s） | **2.26 m/s** | 最低点，−64% |
| 拐角后 1.5 s | 5.5 m/s | 恢复 |

拐角处 x 仅移动 0.4 m、y 移动 0.7 m ⇒ 实际转弯半径 ≈ **0.5 m**，

```text
a_lat = v²/R = 2.26² / 0.5 ≈ 10.2 m/s²   （≈1.0 g）
```

若要保持 6 m/s 过同一 90° 弯：

```text
R = v²/a_lat = 36 / 10 ≈ 3.6 m    或   a_lat = 72 m/s² ≈ 7.3 g（不可能）
```

**因此“拐角降到 2–2.5 m/s”是几何与动力学下的唯一可行解**，不是策略缺陷。真正的问题是：现有指令是“目标点差分”

```python
command_velocity = (tar_pos(t+1) − tar_pos(t)) / dt   # 大小恒定，方向在顶点瞬变
```

策略**看不到**转角信息，只能事后修正；而速度项 `exp(−k‖v−v_cmd‖²)` 会把这段“必须的转弯”记成**跟踪误差**——正确行为被惩罚。

> 也正因如此，§3 里“拐角速度掉到 0.01 m/s”的判定需要按本节口径重测：**拐角慢 ≠ 停滞**，停滞应当定义为“违抗指令”（见 10.5）。

### 10.2 方案：预瞄（pure pursuit）+ 双指令跟踪

不改网格、不改路线 JSON，只改“怎么取目标”和“怎么产生指令”。

```text
① 预瞄点      P_L = path(s + L)，s 为当前弧长，L = 预瞄距离
② 方位角      α   = angle(P_L − robot) 在机体坐标系下的偏角
③ 角速度指令  ω_cmd = 2 · v · sin(α) / L          （纯追踪转向律, Coulter 1992）
④ 等效果曲率  κ_eff = 2 sin(α) / L
⑤ 线速度上限  v_cmd = min( v_nominal , sqrt(a_lat_max / κ_eff) )
```

- 直道：α≈0 ⇒ ω_cmd≈0，v_cmd = v_nominal；
- 90° 弯、L = 1.5 m：R_eff ≈ L/(2 sin45°) ≈ **1.06 m**，`v_cap = √(2.0×1.06) ≈ 1.46 m/s`，`ω_cmd ≈ 1.38 rad/s`；
- **终点**：`v_cmd = ω_cmd = 0`，即“停下 = 跟踪零指令”，与 Stage 6 的站稳判据天然一致。

**预瞄距离与走廊宽度的匹配**（切角 d = R_eff(1 − cos(θ/2))）：

| 走廊净宽 | 建议 L | 90° 切角 | 备注 |
|---|---:|---:|---|
| ≥ 2.0 m | 1.5 m | ≈0.31 m | 安全 |
| 1.2 – 2.0 m | 1.0 m | ≈0.14 m | 推荐窄道使用 |
| < 1.2 m | 0.6 m 或关闭预瞄 | ≈0.04 m | 改用几何圆角（§1.1 已要求圆弧） |

> 注意：§1.1 已规定“转弯几何 = 圆弧 R ∈ [0.9, 2.0]”。**几何圆弧与预瞄并不冲突**——圆弧保证 κ 有界，预瞄保证指令连续且不滞后（纯低通滤波会引入相位滞后，预瞄不会）。

### 10.3 奖励项

```python
r_lin = exp(-||v_xy - v_cmd||² / sigma_v)        # sigma_v = 0.25 (m/s)²
r_yaw = exp(-(omega_z - omega_cmd)² / sigma_w)   # sigma_w = 0.25 (rad/s)²
```

权重（占 task 部分，取代现有 `velocity_reward_weight: 0.35`）：

| 项 | 权重 | 说明 |
|---|---:|---|
| `r_lin` | **0.50** | 主项，替代原 `velocity` 项 |
| `r_yaw` | **0.20** | 新增，唯一负责转向的项 |
| `location` | 0.25（原 1.3 权重下调） | 仅作弱引导，避免“瞬间转向”再次被位置项主导 |
| `power` | −0.02 | 不变 |
| `safety`（swing/landing/impact，clip ±0.30） | 不变 | 不变 |
| task / AMP | 0.7 / 0.3（按 §2） | 不变 |

对应新增配置：

```yaml
enable_velocity_reward: true
velocity_reward_weight: 0.50        # r_lin
enable_yaw_rate_reward: true
yaw_rate_reward_weight: 0.20        # r_yaw
sigma_v: 0.25
sigma_omega: 0.25
lookahead_distance: 1.5             # 按 10.2 的表选
a_lat_max: 2.0                      # m/s²，曲率限速
v_nominal: 2.5                      # m/s，直道巡航（原实测 6.2 偏快）
```

### 10.4 观测（必须加，否则策略“看不见”转角）

在观测末尾追加 4 维：

```text
[v_cmd 在机体坐标系的 x] , [v_cmd 在机体坐标系的 y] , [omega_cmd] , [到下一个拐角的弧长 / 5.0]
```

依据：Rudin 2022 与 Margolis 2022 都把指令（vx, vy, ωz）放进观测；否则策略只能事后修正，无法预判。

> 副作用：观测维度变化，旧 checkpoint 的 actor 首层不兼容。处理方式二选一：
> ① 从 stage6 权重热启动并重新初始化 actor 第一层（推荐）；
> ② 先不加观测（退化版），只改奖励，确认拐角跟踪误差下降后再加入。

### 10.5 停滞判定改为“违抗指令”（取代 §3 的绝对速度口径）

```python
stall = (||v_cmd|| > 0.5) & (||v_xy|| < 0.3)      # 持续 N 帧
penalty = -W_STALL * warmup * clip(stall_count / N, 0, 1)
```

- 转弯、避让、终点减速时 `v_cmd` 本身就低 ⇒ **永不误罚**；
- 只有“指令要求走、实际站着”才算停滞；
- §3 的 `stall_penalty_weight: 0.15` 与预热机制保留，只把门控从“目标隐含速度”换成 `v_cmd`。

### 10.6 验收指标（在 §6 基础上新增）

| 指标 | 目标 |
|---|---|
| 线速度跟踪 RMSE（全轨迹） | < 0.35 m/s |
| **角速度跟踪 RMSE（拐角 ±1 s）** | **< 0.40 rad/s** |
| 拐角实测速度 / v_cmd | 0.8 – 1.2（不再出现 6.2→2.3 的失控式下跌） |
| 终点终止速度 / 躯干倾角 | < 0.5 m/s / < 20°（沿用 Stage 6 判据） |
| 路线完成率（4 块 route mesh） | ≥ 3/4 |

### 10.7 文献依据

| 环节 | 依据 |
|---|---|
| 线速度 + 角速度双指令、指数核 σ=0.25 | Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2022 |
| twist 指令 + 指令进观测 | Margolis & Agrawal, *Walk These Ways*, CoRL 2022 |
| 指令跟踪型主奖励（含偏航率） | Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science Robotics 2019 |
| 复杂地形仍以速度/偏航率跟踪为主项 | Lee et al., *Learning quadrupedal locomotion over challenging terrain*, Science Robotics 2020 |
| 人形复杂地形速度跟踪 | Radosavovic et al., *Learning Humanoid Locomotion over Challenging Terrain*, 2024 |
| 预瞄 / 纯追踪转向律 | Coulter, *Implementation of the Pure Pursuit Path Tracking Algorithm*, CMU-TR 1992 |
| 楼梯上全向指令（vx, vy, ω） | *Omnidirectional Humanoid Locomotion on Stairs via Unsafe Stepping Penalty and Sparse LiDAR Elevation Mapping*, arXiv:2603.07928 |

### 10.8 实施清单（对应 §4 的补充）

| # | 文件 | 改动 | 工作量 |
|---|---|---|---|
| 1 | `pacer/env/util/traj_generator.py` | 弧长参数化 + 曲率 + `calc_cmd()`（v_cmd/ω_cmd/预瞄点） | 中 |
| 2 | `pacer/env/tasks/humanoid_pedestrain_terrain.py` | `command_velocity` → `(v_cmd, ω_cmd)`；新增 `r_yaw`；观测追加 4 维；停滞判定改用 `v_cmd` | 中 |
| 3 | `pacer/env/util/fixed_path.py` | `apply_to_verts` 支持圆弧段（配合 §1.1） | 中 |
| 4 | 配置文件 | 10.3 的新增项 | 小 |
| 5 | `script/pitch_timeline.py`、新增 `speed_track_check.py` | 拐角速度/角速度跟踪误差统计 | 小 |

**训练顺序建议**：Stage 7 前 100 轮先只跑“单层平地 + 1 个转弯”（§5.1 已规定），用这段验证 `r_yaw` 是否生效（角速度跟踪 RMSE 应在前 100 轮内降到 0.6 rad/s 以下），再进入楼梯段。

