#!/usr/bin/env python
"""
Fix PyTorch loading for PyTorch 2.6+ by monkey patching torch.load
to use weights_only=False by default.
"""
import torch
import sys
import argparse
import subprocess
import torch.serialization

# Add argparse.Namespace to safe globals
torch.serialization.add_safe_globals([argparse.Namespace])

# Monkey patch torch.load to use weights_only=False
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

# If script is run directly, execute admet_predict with remaining args
if __name__ == "__main__":
    cmd = ["admet_predict"] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
