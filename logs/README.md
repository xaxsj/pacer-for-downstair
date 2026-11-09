# PACER 训练崩溃 —— 复现记录与结论

服务器：`hoi`（211.81.55.164，user `ljx`），仓库 `/data/ljx/code/pacer-main`
日期：2026-11-05（UTC）
执行工具：`script/pacer_train_probe.sh`（服务器端）+ `script/pacer_train_probe.ps1`（本地驱动）

---

## 一、三次运行对照

| 运行 | envs | iters | 开始 | 结束 | 耗时 | 结果 |
|---|---|---|---|---|---|---|
| 原始训练（ljx 发起） | 1024 | 500 | 17:22:29 | 17:31:54 | 9m25s | **SIGSEGV (139)** |
| probe_small（我第一次探针） | 128 | 5 | 17:52:33 | 17:57:09 | 4m36s | **SIGKILL (137)** |
| **probe32（本次复现）** | **32** | **2** | 17:58:55 | **18:07:10** | **8m15s** | **SIGSEGV (139)** |

对应日志：

- `pacer_stage1_train_crashed_17h22.log` —— 原始训练（1024 envs）
- `pacer_probe_small_launch.log` —— 128 envs 那次（被 SIGKILL）
- `pacer_probe32.log` —— 本次 32 envs 复现（含完整前后环境信息）

---

## 二、四条确定结论

### 1. 崩溃与 env 数量无关

**32 个 env 一样崩，耗时 8m15s，和 1024 env 的 9m25s 几乎一样。**
→ 不是规模问题，也不是显存/内存不足。

### 2. 崩溃发生在**第一次迭代完成之前**

`print_stats` 已通过临时配置（不改源码）改成 `True`（两处，agent + AMP），但整段 8 分钟里：

- **没有任何一行 `Ep:`**（训练迭代统计）
- **没有 `MAX EPOCHS NUM!`**

日志顺序是：
```
Started to train
Found checkpoint output/exp/pacer_probe32/Humanoid.pth
100%|...| 50/50 [00:00<00:00, 1537.67it/s]     <- 地形生成
（静默 8 分钟）
Segmentation fault
```

→ 那 8 分钟**不是训练**，而是**地形之后的剩余环境构建 / 首个 play_steps**。

### 3. 不是 OOM

`free -g`：503 GB 总量 / **379 GB 可用**；系统无 OOM 记录。

### 4. 可稳定复现

三次里有两次（1024 / 32 envs）都是以相同方式 SIGSEGV。

---

## 三、根因（已定位）

用 `PYTHONFAULTHANDLER=1`（只加环境变量，未改源码）拿到崩溃栈：

```
Fatal Python error: Segmentation fault

Current thread:
  humanoid_pedestrain_terrain.py:828  in create_training_ground    <- 崩在这里
  humanoid_pedestrain_terrain.py:771  in _create_ground_plane
  humanoid.py:434                     in create_sim
  base_task.py:113                    in __init__
  ...
  run.py:290                          in main
```

第 828 行：

```python
self.gym.add_triangle_mesh(self.sim,
                           self.terrain.vertices.flatten(order='C'),
                           self.terrain.triangles.flatten(order='C'),
                           tm_params)
```

崩溃发生在 **Isaac Gym / PhysX 的原生调用 `add_triangle_mesh`**，所以是 C++ 层
SIGSEGV，Python 层没有任何异常。

### 网格规模计算

`Terrain.__init__`（同文件 1082 行起）把 `numLevels × numTerrains` 块地形拼成**一张**
大 heightfield，分辨率 `horizontal_scale = 0.1`（0.1 m/像素），四周再加 `border = 500` 像素：

```
env_rows = numLevels        env_cols = numTerrains
tot_rows = numLevels    * (mapLength / 0.1) + 2*500
tot_cols = numTerrains  * (mapWidth  / 0.1) + 2*500
triangles ≈ 2 * (tot_rows - 1) * (tot_cols - 1)
```

| 配置 | numLevels | numTerrains | heightfield | **三角面数** | 结果 |
|---|---|---|---|---|---|
| `pacer.yaml`（上游默认） | 5 | 20 | 2000 × 5000 | **20.0 M** | 可用 |
| `pacer_stage1_stairs.yaml`（ljx） | 5 | **50** | 2000 × 11000 | **44.0 M** | **段错误** |

**结论：`numTerrains: 50` 让地形三角面数从 2000 万涨到 4400 万，超出 PhysX 单次
`add_triangle_mesh` 的承受范围，直接段错误。**

这也解释了那 8 分钟：`convert_heightfield_to_trimesh` 在 2200 万格的网格上转换本身就要
这么久，转完交给 PhysX 时立刻崩 —— 所以日志停在地形 tqdm 之后、没有任何迭代输出。

### 修复建议

1. **`numTerrains: 50` → `20`**（对齐上游默认，4400 万 → 2000 万三角面）
2. 或减小 `mapLength` / `mapWidth`（当前 20 m，每块 200×200 像素）
3. 或启用 `small_terrain`（代码里会把 mapLength/Width 压到 8 m）
4. 若要保留 50 块地形的多样性，可改为**分批建地形**或减小单块尺寸，
   而不是把 50 块拼进同一张网格

**验证方式**：把 `numTerrains` 改成 20 重跑同一条命令，若能越过
`add_triangle_mesh` 并打印出 `Ep:`，即证实。

---

## 四、顺带发现的两个配置问题

### 1. `--max_iterations` 与 resume 的 epoch 冲突

`pacer/env/config.py:125`：

```python
if args.max_iterations > 0:
    cfg_train["params"]["config"]['max_epochs'] = args.max_iterations   # 500
```

而 resume 的检查点 **`epoch = 41250, frame = 2027520000`**（`output/release/pacer/Humanoid.pth` 的发布版权重）。

训练循环判断：`if epoch_num > self.max_epochs: save(); print('MAX EPOCHS NUM!'); return`

若 `epoch_num` 从 41250 续算，则 `41250 > 500` → **第一轮就存盘退出**，不会真的微调 500 轮。
（此点尚未实证，因为训练在第一轮之前就崩了。）

### 2. 训练脚本没有 Xvfb

| 脚本 | Xvfb |
|---|---|
| `scripts/pacer_run.sh`（推理） | 有，自动选 :90-:99 |
| `scripts/pacer_train_stage1.sh`（训练） | **无** |

但注意：13:01/13:02 的段错误发生在**带 Xvfb** 的 `pacer_run.sh` 里，所以"缺 Xvfb"**不是**本次崩溃的原因，只是两个脚本的不一致之处。

---

## 五、服务器环境注意

- 8 张 4090D 常年 90-99% 占用（多人共用）
- 存在 `zcx` 的 `watchdog.py`（`/home/zcx/.local/lib/8x4090d-proxy/watchdog.py`）
- 我的 128-env 探针在 4m36s 时被 **SIGKILL**，怀疑与该策略有关（未证实）
- 因此在这台机器上跑训练，随时可能被外部杀掉；**SIGKILL(137) ≠ SIGSEGV(139)**，不要把两者混为一谈
