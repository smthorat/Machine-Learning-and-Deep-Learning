#!/usr/bin/env python
"""
Find and patch all torch.load calls in the Chemprop library to use weights_only=False
"""
import os
import glob
import re
import sys
import site
from pathlib import Path

def find_chemprop_files():
    """Find all Python files in the Chemprop package directory."""
    # Find the site-packages directory
    site_packages = site.getsitepackages()[0]
    print(f"Looking for Chemprop in {site_packages}")
    
    # Find all Python files in the Chemprop directory
    chemprop_dir = os.path.join(site_packages, "chemprop")
    if not os.path.exists(chemprop_dir):
        print(f"Chemprop directory not found at {chemprop_dir}")
        return []
    
    py_files = glob.glob(f"{chemprop_dir}/**/*.py", recursive=True)
    print(f"Found {len(py_files)} Python files in Chemprop")
    return py_files

def patch_file(file_path):
    """Patch torch.load calls in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match torch.load calls without weights_only parameter
    pattern = r'torch\.load\s*\(([^)]*)\)'
    
    # Check if we need to modify this file
    if 'torch.load' in content:
        # Function to process each match
        def replace_torch_load(match):
            args = match.group(1).strip()
            if 'weights_only' not in args:
                if args.endswith(')'):
                    # This shouldn't happen but handle it just in case
                    return f'torch.load({args[:-1]}, weights_only=False)'
                elif not args:
                    return 'torch.load(weights_only=False)'
                else:
                    return f'torch.load({args}, weights_only=False)'
            return match.group(0)
        
        # Replace all torch.load calls
        new_content = re.sub(pattern, replace_torch_load, content)
        
        # Only write back if changes were made
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            return True
    
    return False

def main():
    """Find and patch all Chemprop files."""
    py_files = find_chemprop_files()
    patched_files = 0
    
    for file_path in py_files:
        if patch_file(file_path):
            patched_files += 1
            print(f"Patched {file_path}")
    
    print(f"Patched {patched_files} files")
    
    # Also check admet_ai package
    site_packages = site.getsitepackages()[0]
    admet_dir = os.path.join(site_packages, "admet_ai")
    if os.path.exists(admet_dir):
        print(f"Found admet_ai package at {admet_dir}")
        admet_files = glob.glob(f"{admet_dir}/**/*.py", recursive=True)
        admet_patched = 0
        
        for file_path in admet_files:
            if patch_file(file_path):
                admet_patched += 1
                print(f"Patched {file_path}")
        
        print(f"Patched {admet_patched} files in admet_ai package")

if __name__ == "__main__":
    main()