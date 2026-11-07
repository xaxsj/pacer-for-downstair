#!/usr/bin/env bash
# PACER conda environment on nmb2 (RTX 3080 Ti / 12 GB)
#
# usage:
#   source /root/autodl-tmp/pacer/pacer_env.sh
#   python pacer/run.py ...
#
# 关键点：
#   - conda 环境不在 /root/miniconda3/envs，而在 /root/autodl-tmp/envs/pacer（56 G 盘）
#   - python 3.8 是硬性要求：Isaac Gym 只提供 gym_36/37/38.so 预编译绑定
#   - env 的 bin 必须进 PATH，否则 torch 找不到 pip 装的 ninja，JIT 扩展编译会失败
#   - env 的 lib 必须进 LD_LIBRARY_PATH，否则 gym_38.so 找不到 libpython3.8.so.1.0

export PACER_ENV=/root/autodl-tmp/envs/pacer
export PACER_ROOT=/root/autodl-tmp/pacer/repo
export ISAACGYM_PY=/root/autodl-tmp/isaacgym_pkg/isaacgym/python

# 1) conda 环境自带二进制（python / pip / ninja / ninja-build）
export PATH="$PACER_ENV/bin:$PATH"

# 2) libpython3.8.so.1.0（gym_38.so 依赖）+ CUDA 11.8 运行时
export LD_LIBRARY_PATH="$PACER_ENV/lib:/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 3) Isaac Gym python 包 + PACER 包
export PYTHONPATH="$ISAACGYM_PY:$PACER_ROOT/pacer${PYTHONPATH:+:$PYTHONPATH}"

# 4) torch 扩展编译缓存（避免每次都重建 gymtorch）
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/tmp/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

# 5) 单卡：默认用 GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 6) 无头运行时常量
export PYTHONUNBUFFERED=1
