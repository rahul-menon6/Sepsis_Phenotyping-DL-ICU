"""
Normalizes demographic features and splits the preprocessed dataset into
train/val/test sets following Hilders et al. (2026): 70/15/15 patient-level split.

Input:  data/features/features_processed.csv
Output: data/splits/train.csv, val.csv, test.csv

Usage:
    conda activate ml-env
    python src/04_split_and_normalize.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'features_processed.csv')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'splits')
os.makedirs(output_dir, exist_ok=True)

print('Loading preprocessed features...')
df = pd.read_csv(input_path)
print(f'  {df.admissionid.nunique()} admissions, {len(df)} rows')

# ── Age normalization ─────────────────────────────────────────────────────────
# Hilders et al.: age grouped into 7 bins, mapped to -1.0 to +1.0

age_norm_map = {
    '18-39': -1.0,
    '40-49': -0.6667,
    '50-59': -0.3333,
    '60-69':  0.0,
    '70-79':  0.3333,
    '80+':    1.0,
}
df['admission_age_norm'] = df['agegroup'].map(age_norm_map)

# ── Weight normalization ──────────────────────────────────────────────────────
# Hilders et al.: clip to [55, 115], scale to [-1, 1]
# weightgroup midpoints: '59-'→52, '60-69'→65, '70-79'→75, '80-89'→85,
#                        '90-99'→95, '100-109'→105, '110+'→115

weightgroup_midpoint = {
    '59-':     52.0,
    '60-69':   65.0,
    '70-79':   75.0,
    '80-89':   85.0,
    '90-99':   95.0,
    '100-109': 105.0,
    '110+':    115.0,
}
df['weight_mid'] = df['weightgroup'].map(weightgroup_midpoint)
df['weight_null'] = np.where(df['weight_mid'].isna(), 1, 0)

def scale(val, min_val=55, max_val=115, min_range=-1, max_range=1):
    normalized = (val - min_val) / (max_val - min_val)
    return normalized * (max_range - min_range) + min_range

df['weight_norm'] = scale(df['weight_mid'].clip(55, 115))
df['weight_norm'] = df['weight_norm'].fillna(0.0)

# height not available in AmsterdamUMCdb v1.0.2 cohort extraction
# add placeholder binary flag as Hilders did
df['height_null'] = 1
df['height_norm'] = 0.0

# ── Gender encoding ───────────────────────────────────────────────────────────
# already binary (1=male, 0=female) from feature extraction
# confirm and fill any remaining nulls
df['gender'] = df['gender'].fillna(0).astype(int)

# ── Drop raw categorical columns ──────────────────────────────────────────────
df.drop(columns=['agegroup', 'weightgroup', 'weight_mid'], inplace=True)

print(f'  Columns after normalization: {len(df.columns)}')

# ── Train/val/test split (70/15/15, patient-level) ────────────────────────────

print('Splitting into train/val/test...')
admission_ids = df['admissionid'].unique()

train_ids, temp_ids = train_test_split(admission_ids, test_size=0.30, random_state=42)
val_ids, test_ids   = train_test_split(temp_ids,      test_size=0.50, random_state=42)

train = df[df['admissionid'].isin(train_ids)]
val   = df[df['admissionid'].isin(val_ids)]
test  = df[df['admissionid'].isin(test_ids)]

print(f'  Train: {train.admissionid.nunique()} admissions, {len(train)} rows')
print(f'  Val:   {val.admissionid.nunique()} admissions, {len(val)} rows')
print(f'  Test:  {test.admissionid.nunique()} admissions, {len(test)} rows')

# ── Save ──────────────────────────────────────────────────────────────────────

train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val.to_csv(os.path.join(output_dir, 'val.csv'),     index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'),   index=False)

print(f'\nSaved splits to {output_dir}')
print(f'Final columns: {list(df.columns)}')
