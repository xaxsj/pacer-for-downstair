"""Validate the four stage configs and print their terrain curriculum."""
import glob

import yaml

for path in sorted(glob.glob("pacer/data/cfg/pacer_stage*_stairs.yaml")):
    env = yaml.safe_load(open(path, encoding="utf-8"))["env"]
    t = env["terrain"]
    print(f"{path.split('/')[-1]:32s} "
          f"riser {t['stair_riser_min']:.2f}-{t['stair_riser_max']:.2f}  "
          f"tread {t['stair_tread_min']:.2f}-{t['stair_tread_max']:.2f}  "
          f"W {t['stair_width_bins']}  "
          f"steps {t['stair_up_steps_min']}-{t['stair_up_steps_max']}  "
          f"loc {env['location_reward_weight']}  "
          f"bonus {env['success_bonus']}  pen {env['off_route_penalty']}")
