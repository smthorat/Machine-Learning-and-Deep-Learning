#!/usr/bin/env python
"""Create larger files by repeating the 100-molecule file."""
import pandas as pd
import os
import sys

print("Creating larger molecule files...")

# Load the 100-molecule file
print("Loading 100-molecule file...")
df_100 = pd.read_csv('/Volumes/Jagannath/Projects/ADMET AI/admet_ai/data/drugbank/drugbank_approved_100.csv')
print(f"Loaded file with {len(df_100)} molecules and columns: {', '.join(df_100.columns)}")

# Ensure the output directory exists
output_dir = 'data/drugbank'
os.makedirs(output_dir, exist_ok=True)

# Create the 1000-molecule file
print("Creating 1000-molecule file...")
df_1000 = pd.concat([df_100] * 10, ignore_index=True)
df_1000.to_csv(os.path.join(output_dir, 'drugbank_approved_1000.csv'), index=False)
print(f"Created file with {len(df_1000)} molecules")

# Ask before creating the 1M file since it could be large
create_1m = input("Create 1M molecule file? This will be a large file! (y/n): ").lower().strip() == 'y'

if create_1m:
    print("Creating 1M-molecule file...")
    # We'll do this in chunks to avoid memory issues
    chunk_size = 100  # Process 100k molecules at a time
    file_path = os.path.join(output_dir, 'drugbank_approved_1M.csv')
    with open(file_path, 'w') as f:
        # Write header
        f.write(','.join(df_100.columns) + '\n')
        
        # Write data in chunks
        for i in range(0, 1000, chunk_size):
            print(f"Processing chunk {i//chunk_size + 1} of {1000//chunk_size}...")
            # Create a chunk of 100k molecules (100 * chunk_size)
            chunk = pd.concat([df_100] * (chunk_size * 10), ignore_index=True)
            # Append to file without header
            chunk.to_csv(f, header=False, index=False, mode='a')
    
    # Verify file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"Created 1M-molecule file (size: {file_size:.1f} MB)")
else:
    print("Skipping 1M-molecule file creation")

print("Done!")