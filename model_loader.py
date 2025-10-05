"""Helper to load generator weights robustly on CPU/GPU and report mismatches."""
import torch
import sys




def load_model_state(model, path, device=None):
"""Load state dict into model with safe map_location and informative errors.


Returns True if load succeeded, False otherwise.
"""
if device is None:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
map_loc = torch.device(device)
try:
state = torch.load(path, map_location=map_loc)
except Exception as e:
print(f"ERROR: could not torch.load('{path}') -> {e}")
return False


# If file saved as dict with ['model_state_dict'] wrapper, try common keys
if isinstance(state, dict):
# heuristics: look for 'state_dict' keys
candidate = None
for key in ('state_dict', 'model_state_dict', 'generator_state_dict', 'netG_state_dict'):
if key in state:
candidate = state[key]
break
if candidate is not None:
state = candidate


# attempt load
try:
missing, unexpected = model.load_state_dict(state, strict=False)
except Exception as e:
# fallback: if state is a dict with nested keys try to extract
try:
model.load_state_dict(state, strict=False)
return True
except Exception as e2:
print("FATAL: model.load_state_dict failed ->", e2)
return False


# report problems
if hasattr(missing, '__len__') and (len(missing) or len(unexpected)):
print(f"Warning: missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
if len(missing):
print("Missing:\n", missing[:10])
if len(unexpected):
print("Unexpected:\n", unexpected[:10])
return True




if __name__ == '__main__':
print('Run this as a library, not standalone.')
