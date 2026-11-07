#!/usr/bin/env python3
"""Run process_amass_db.py with its debugging breakpoints disabled."""
import ipdb
import runpy
import sys

ipdb.set_trace = lambda *args, **kwargs: None
sys.argv = ["uhc/data_process/process_amass_db.py"] + sys.argv[1:]
runpy.run_path("uhc/data_process/process_amass_db.py", run_name="__main__")
