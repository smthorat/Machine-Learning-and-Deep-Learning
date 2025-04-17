#!/usr/bin/env python
"""Replace specific compounds in the 100-molecule file with alternatives."""
import pandas as pd

# Load the files
print("Loading files...")
df_100 = pd.read_csv('/Volumes/Jagannath/Projects/ADMET AI/admet_ai/data/drugbank/drugbank_approved_100.csv')
df_full = pd.read_csv('/Volumes/Jagannath/Projects/ADMET AI/admet_ai/data/drugbank/drugbank_approved.csv')

# Define replacements
replacements = {
    'Helium': 'Cabazitaxel',
    'Chromic nitrate': 'Butorphanol',
    'Perboric acid': 'Methazolamide',
    'Fluoride ion F-18': 'Tetracaine'
}

# Check if the compounds to replace exist
compounds_to_replace = []
for old_name in replacements.keys():
    if old_name in df_100['name'].values:
        compounds_to_replace.append(old_name)
        print(f"Found {old_name} to replace")
    else:
        print(f"Warning: {old_name} not found in the 100-molecule file")

# Check if replacement compounds exist
for new_name in replacements.values():
    if new_name not in df_full['name'].values:
        print(f"Warning: Replacement compound {new_name} not found in the full database")

# Perform replacements
replacement_count = 0
for old_name, new_name in replacements.items():
    if old_name in df_100['name'].values and new_name in df_full['name'].values:
        # Find the row index in the 100-molecule file
        idx = df_100[df_100['name'] == old_name].index[0]
        
        # Get the replacement compound data
        replacement = df_full[df_full['name'] == new_name].iloc[0]
        
        # Replace the values
        for col in df_100.columns:
            df_100.at[idx, col] = replacement[col]
        
        replacement_count += 1
        print(f"Replaced {old_name} with {new_name}")

# Save the updated file
df_100.to_csv('/Volumes/Jagannath/Projects/ADMET AI/admet_ai/data/drugbank/drugbank_approved_100.csv', index=False)
print(f"Replacements complete. Replaced {replacement_count} compounds.")