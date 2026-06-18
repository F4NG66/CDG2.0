#!/usr/bin/env python3
"""p3_probe.py — Linear Probe Sweep. Delegates to s2_probe_sweep logic."""
import os, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
subprocess.run([sys.executable, "scripts/s2_probe_sweep.py"] + sys.argv[1:])
