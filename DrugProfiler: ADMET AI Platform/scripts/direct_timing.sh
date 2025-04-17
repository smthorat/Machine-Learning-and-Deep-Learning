#!/bin/bash
# Run timing tests directly after patching Chemprop

# First patch Chemprop
echo "Patching Chemprop to fix torch.load issues..."
python patch_chemprop.py

# Run timing tests without any wrapper
for NUM_MOLECULES in 1 10 100 1000
do
  for ITER in 1 2 3
  do
    echo "-------------------------------------------------------------------------"
    echo "Timing ADMET-AI on ${NUM_MOLECULES} molecules, iteration ${ITER}"
    echo "-------------------------------------------------------------------------"
    time admet_predict \
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
    time admet_predict \
        --data_path data/drugbank/drugbank_approved_1M.csv \
        --save_path data/drugbank/drugbank_approved_1M_admet_${ITER}.csv \
        --models_dir models/tdc_admet_all_multitask/chemprop_rdkit \
        --smiles_column smiles
    echo ""
  done
fi

echo "All timing tests completed!"