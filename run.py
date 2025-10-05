"""Launcher for Lunar-PSR-Enhancement


Usage:
python run.py test # run testing.py (inference) with robust model loading
python run.py train # start a short smoke-training run
python run.py gui # launch GAN_GUI.py


This script sets PYTHONPATH to repo root so imports resolve and wraps torch model loads.
"""
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


import argparse
import importlib


from model_loader import load_model_state




def safe_run(module_name, func_name='main'):
try:
mod = importlib.import_module(module_name.replace('.py',''))
except Exception as e:
print(f"ERROR: failed to import {module_name}: {e}")
return 1
# prefer an explicit entrypoint if present
if hasattr(mod, func_name):
try:
getattr(mod, func_name)()
return 0
except TypeError:
# maybe script guarded by if __name__ == '__main__'
return os.system(f"python {module_name}")
except Exception as e:
print(f"ERROR running {module_name}.{func_name}: {e}")
return 1
else:
# fallback: execute as script
return os.system(f"python {module_name}")




def preflight_check():
import torch
print('Python:', sys.version.splitlines()[0])
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
# quick check that generator_final.pth exists
gen = ROOT / 'generator_final.pth'
if not gen.exists():
print('\nWARNING: generator_final.pth not found in repo root — testing.py may fail.')
else:
print('Found generator_final.pth')




def main():
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['test','train','gui'], help='what to run')
args = parser.parse_args()


preflight_check()
sys.exit(main())
