#!/usr/bin/env python
"""
Create a 1000-molecule file by repeating the 100-molecule file 10 times.
"""
import pandas as pd
import os

# File paths
input_file = 'data/drugbank/drugbank_approved_100.csv'
output_file = 'data/drugbank/drugbank_approved_1000.csv'

print(f"Creating {output_file} from {input_file}...")

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} not found!")
    exit(1)

# Load the 100-molecule file
try:
    df_100 = pd.read_csv(input_file)
    print(f"Loaded {len(df_100)} molecules from {input_file}")
except Exception as e:
    print(f"Error loading {input_file}: {e}")
    exit(1)

# Create 1000-molecule dataframe by repeating 10 times
df_1000 = pd.concat([df_100] * 10, ignore_index=True)
print(f"Created dataframe with {len(df_1000)} molecules")

# Save to file
try:
    df_1000.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df_1000)} molecules to {output_file}")
except Exception as e:
    print(f"Error saving to {output_file}: {e}")
    exit(1)

print("Done!")