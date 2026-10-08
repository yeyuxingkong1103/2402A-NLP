# -*- coding: utf-8 -*-
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(HERE / "batch_shots.py")], check=True)
