python scripts/prepare_tdc_admet_group.py \
    --raw_data_dir data/tdc_admet_group_raw \
    --save_dir data/tdc_admet_group


python scripts/prepare_tdc_admet_all.py \
    --save_dir data/tdc_admet_all \
    --skip_datasets herg_central hERG_Karim ToxCast 


python scripts/merge_tdc_admet_multitask.py \
    --data_dir data/tdc_admet_all \
    --save_dir data/tdc_admet_all_multitask


python scripts/merge_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_path data/tdc_admet_all.csv


python scripts/compute_rdkit_features.py \
    --data_dir data/tdc_admet_group \
    --smiles_column Drug


python scripts/compute_rdkit_features.py \
    --data_dir data/tdc_admet_all \
    --smiles_column smiles


python scripts/compute_rdkit_features.py \
    --data_dir data/tdc_admet_all_multitask \
    --smiles_column smiles


# Train Chemprop-RDKit ADMET predictors

python scripts/train_tdc_admet_group.py \
    --data_dir data/tdc_admet_group \
    --save_dir models/tdc_admet_group \
    --model_type chemprop_rdkit


python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir models/tdc_admet_all \
    --model_type chemprop_rdkit



python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all_multitask \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop_rdkit


# Evaluate TDC ADMET Benchmark Group models

python scripts/evaluate_tdc_admet_group.py \
    --data_dir data/tdc_admet_group_raw \
    --preds_dir models/tdc_admet_group/chemprop_rdkit



python scripts/get_drugbank_approved.py \
    --data_path data/drugbank/drugbank.xml \
    --save_path data/drugbank/drugbank_approved.csv



admet_predict \
    --data_path data/drugbank/drugbank_approved.csv \
    --save_path data/drugbank/drugbank_approved_physchem_admet.csv \
    --models_dir models/tdc_admet_all_multitask/chemprop_rdkit \
    --smiles_column smiles


for NUM_MOLECULES in 1 10 100
do
python scripts/sample_molecules.py \
    --data_path data/drugbank/drugbank_approved.csv \
    --num_molecules ${NUM_MOLECULES} \
    --max_smiles_length 200 \
    --save_path data/drugbank/drugbank_approved_${NUM_MOLECULES}.csv
done



TDC ADMET Results.xlsx

python scripts/fixed_plot_tdc_results.py     --results_path results/tdc_results/Results.csv     --save_dir plots/tdc_results

python scripts/plot_drugbank_approved.py \
    --data_path data/drugbank/drugbank_approved.csv \
    --save_dir plots/drugbank_approved



python scripts/plot_admet_speed.py \
    --results_path results/ADMET\ Speed\ Comparison.xlsx \
    --save_path plots/admet_speed.pdf