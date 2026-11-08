"""Generate the Stage 2-4 configs from the Stage 1 stairs template.

Usage: python script/make_stage_configs.py

Reward and termination settings are copied unchanged (doc §8: the reward is
fixed after Stage 1); only the terrain curriculum differs.
"""
import re

SRC = "pacer/data/cfg/pacer_stage1_stairs.yaml"

STAGES = {
    2: dict(riser=(0.08, 0.15), tread=(0.31, 0.40),
            width_bins=[3.0, 2.8, 2.4, 2.0, 1.8],
            width_probs=[0.10, 0.25, 0.30, 0.20, 0.15],
            up=(6, 9), down=(6, 9), start=(0.35, 0.80)),
    3: dict(riser=(0.10, 0.18), tread=(0.28, 0.38),
            width_bins=[2.8, 2.4, 2.0, 1.8, 1.6],
            width_probs=[0.15, 0.25, 0.25, 0.20, 0.15],
            up=(8, 12), down=(8, 12), start=(0.30, 0.80)),
    4: dict(riser=(0.08, 0.22), tread=(0.26, 0.38),
            width_bins=[3.0, 2.8, 2.4, 2.0, 1.8, 1.6, 1.2],
            width_probs=[0.05, 0.10, 0.20, 0.20, 0.20, 0.15, 0.10],
            up=(8, 12), down=(8, 12), start=(0.30, 0.80)),
}


def fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def build(stage, p):
    text = open(SRC, encoding="utf-8").read()

    def sub(key, value):
        nonlocal text
        pat = re.compile(rf"^(\s*{key}:\s*).*$", re.MULTILINE)
        assert len(pat.findall(text)) == 1, f"{key} matched {len(pat.findall(text))} times"
        text = pat.sub(lambda m: m.group(1) + value, text)

    sub("stair_riser_min", fmt(p["riser"][0]))
    sub("stair_riser_max", fmt(p["riser"][1]))
    sub("stair_tread_min", fmt(p["tread"][0]))
    sub("stair_tread_max", fmt(p["tread"][1]))
    sub("stair_width_bins", "[" + ", ".join(fmt(v) for v in p["width_bins"]) + "]")
    sub("stair_width_probs", "[" + ", ".join(f"{v:.2f}" for v in p["width_probs"]) + "]")
    sub("stair_up_steps_min", str(p["up"][0]))
    sub("stair_up_steps_max", str(p["up"][1]))
    sub("stair_down_steps_min", str(p["down"][0]))
    sub("stair_down_steps_max", str(p["down"][1]))
    sub("stair_start_dist_min", fmt(p["start"][0]))
    sub("stair_start_dist_max", fmt(p["start"][1]))
    sub("notes", f'"curriculum stage {stage}"')

    out = f"pacer/data/cfg/pacer_stage{stage}_stairs.yaml"
    open(out, "w", encoding="utf-8").write(text)
    print(f"wrote {out}: riser {p['riser']} tread {p['tread']} "
          f"W {p['width_bins']} steps {p['up']}/{p['down']}")


for stage, params in STAGES.items():
    build(stage, params)
