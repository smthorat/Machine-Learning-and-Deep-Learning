#!/bin/bash
# Script to run timing tests with PyTorch 2.6 compatibility

# Create and run PyTorch fix script
cat > fix_torch.py << 'EOL'
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
EOL

# Make script executable
chmod +x fix_torch.py

# Run timing tests
for NUM_MOLECULES in 1 10 100 1000
do
  for ITER in 1 2 3
  do
    echo "-------------------------------------------------------------------------"
    echo "Timing ADMET-AI on ${NUM_MOLECULES} molecules, iteration ${ITER}"
    echo "-------------------------------------------------------------------------"
    time python fix_torch.py \
        --data_path data/drugbank/drugbank_approved_${NUM_MOLECULES}.csv \
        --save_path data/drugbank/drugbank_approved_${NUM_MOLECULES}_admet_${ITER}.csv \
        --models_dir models/tdc_admet_all_multitask/chemprop_rdkit \
        --smiles_column smiles
    echo ""
  done
done

# Ask before running the 1M molecule test
echo "Do you want to run the timing test on 1M molecules? This will take a long time! (y/n)"
read RUN_1M
if [[ "$RUN_1M" == "y" || "$RUN_1M" == "Y" ]]; then
  for ITER in 1 2 3
  do
    echo "-------------------------------------------------------------------------"
    echo "Timing ADMET-AI on 1M molecules, iteration ${ITER}"
    echo "-------------------------------------------------------------------------"
    time python fix_torch.py \
        --data_path data/drugbank/drugbank_approved_1M.csv \
        --save_path data/drugbank/drugbank_approved_1M_admet_${ITER}.csv \
        --models_dir models/tdc_admet_all_multitask/chemprop_rdkit \
        --smiles_column smiles
    echo ""
  done
fi

echo "All timing tests completed!"