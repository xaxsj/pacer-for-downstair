#!/usr/bin/env python3
"""Run convert_amass_isaac.py after supplying the legacy SMPL_Robot alias.

Current smpl_local_robot.py exposes ``Robot``; the older converter imports
``SMPL_Robot``.  Set the alias at runtime instead of editing PACER source.
"""
import os
import runpy
import sys
sys.path.insert(0, os.getcwd())

import uhc.smpllib.smpl_local_robot as slr

if not hasattr(slr, "SMPL_Robot"):
    slr.SMPL_Robot = slr.Robot

sys.argv = ["uhc/data_process/convert_amass_isaac.py"] + sys.argv[1:]
runpy.run_path("uhc/data_process/convert_amass_isaac.py", run_name="__main__")
