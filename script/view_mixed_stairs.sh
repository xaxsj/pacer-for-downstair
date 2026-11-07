#!/bin/bash
# Launch the standalone Isaac Gym viewer for the mixed stair mesh (WSL).
set -u

export PATH=/home/lvjianxiong0505/miniconda3/envs/mimickit/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/lvjianxiong0505/miniconda3/envs/mimickit/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json

cd "$(dirname "$0")/.." || exit 1
exec python script/view_mixed_stairs.py "$@"
