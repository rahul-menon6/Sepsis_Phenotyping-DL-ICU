"""
Preprocesses raw 4-hour windowed features following Hilders et al. (2026):
    1. Outlier removal using physiological thresholds
    2. 24-hour forward-fill (max 6 windows)
    3. Drop features with >60% missingness
    4. Drop patients with >30% missingness
    5. Median imputation
    6. Label derivation (90-day mortality, remaining LOS, next-step RRT, next-step MV)

Input:  data/features/features_raw.csv
Output: data/features/features_processed.csv

Usage:
    conda activate ml-env
    python src/03_preprocess.py
"""

import os
import numpy as np
import pandas as pd

input_path  = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'features_raw.csv')
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'features_processed.csv')

print('Loading features...')
df = pd.read_csv(input_path)
print(f'  {df.admissionid.nunique()} admissions, {len(df)} rows, {len(df.columns)} columns')

# preserve rrt for label derivation before potential feature dropping
rrt_raw = df[['admissionid', 'window', 'rrt']].copy() if 'rrt' in df.columns else None

# Outlier removal 
# Exact thresholds from Hilders et al. data_processing.ipynb features_filter dict

features_filter = {
    'albumin':      (0, 50),
    'alt':          (0, 6000),
    'ast':          (0, 20000),
    'bicarbonate':  (0, 50),
    'bilirubin':    (0, 500),
    'bun':          (0, 100),
    'calcium':      (0, 5),
    'potassium':    (1, 9),
    'chloride':     (50, 125),
    'crp':          (0, 500),
    'creatinine':   (0, 1200),
    'glucose':      (0, 40),
    'heart_rate':   (30, 180),
    'hematocrit':   (0.100, 0.600),
    'hemoglobin':   (0, 15),
    'inr':          (0, 15),
    'lactate':      (0, 30),
    'spo2':         (74, 100),
    'pao2':         (20, 400),
    'paco2':        (15, 150),
    'ph':           (7.0, 8.5),
    'platelet':     (0, 1000),
    'resp_rate':    (5, 50),
    'sodium':       (115, 175),
    'sysbp':        (40, 300),
    'meanbp':       (20, 160),
    'diasbp':       (10, 130),
    'cvp':          (-10, 30),
    'tempc':        (33, 42),
    'wbc':          (0, 100),
    'gcs_eye':      (1, 4),
    'gcs_verbal':   (1, 5),
    'gcs_motor':    (1, 6),
}

print('Applying outlier removal...')
for col, (lo, hi) in features_filter.items():
    if col not in df.columns:
        continue
    df.loc[df[col] < lo, col] = np.nan
    df.loc[df[col] > hi, col] = np.nan

print(f'  Shape after outlier removal: {df.shape}')

# Forward-fill (max 6 windows = 24 hours) 

print('Applying forward-fill...')
df = df.sort_values(['admissionid', 'window']).reset_index(drop=True)

non_fill_cols = ['admissionid', 'window', 'start_time', 'gender', 'agegroup', 'weightgroup']
fill_cols = [c for c in df.columns if c not in non_fill_cols]

df[fill_cols] = df.groupby('admissionid')[fill_cols].ffill(limit=6)
print(f'  Shape after forward-fill: {df.shape}')

# Drop features with >60% missingness 

print('Dropping features with >60% missingness...')
numeric_cols = list(features_filter.keys()) + ['peep', 'vaso_rate', 'fluid_window', 'rrt']
numeric_cols = [c for c in numeric_cols if c in df.columns]

cols_to_drop = []
for col in numeric_cols:
    miss_rate = df[col].isna().sum() / len(df)
    if miss_rate > 0.60:
        cols_to_drop.append(col)

df.drop(columns=cols_to_drop, inplace=True)
numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]
print(f'  Dropped features: {cols_to_drop}')
print(f'  Remaining features: {len(numeric_cols)}')

# Drop patients with >30% missingness 

print('Dropping patients with >30% missingness...')
n_before = df.admissionid.nunique()

missing_per_patient = df.groupby('admissionid')[numeric_cols].apply(
    lambda x: x.isna().sum().sum()
)
total_per_patient = df.groupby('admissionid')[numeric_cols].apply(
    lambda x: x.shape[0] * x.shape[1]
)
missing_pct = missing_per_patient / total_per_patient
patients_to_drop = missing_pct[missing_pct > 0.30].index

df = df[~df['admissionid'].isin(patients_to_drop)]
print(f'  Patients before: {n_before}, after: {df.admissionid.nunique()}')

#  Median imputation 

print('Applying median imputation...')
medians = df[numeric_cols].median()
df[numeric_cols] = df[numeric_cols].fillna(medians)
print(f'  Imputed {len(numeric_cols)} features')

# Label derivation 

print('Deriving labels...')
df['start_time'] = pd.to_datetime(df['start_time'])

# 90-day mortality: did patient die within 90 days after this window's end?
# We use the cohort CSV for death timestamps
cohort_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cohort', 'sepsis3_cohort.csv')
cohort = pd.read_csv(cohort_path)

death_times = cohort[['admissionid', 'dateofdeath']].drop_duplicates()
df = df.merge(death_times, on='admissionid', how='left')
df['dateofdeath'] = pd.to_datetime(df['dateofdeath'], unit='ms', errors='coerce')

window_end = df['start_time'] + pd.Timedelta(hours=4)
time_to_death = df['dateofdeath'] - window_end
df['90day_mortality'] = np.where(
    time_to_death < pd.Timedelta(days=90), 1, 0
)
df.drop(columns=['dateofdeath'], inplace=True)

# Remaining LOS: days remaining in ICU from this window
admit_times = cohort[['admissionid', 'dischargedat']].drop_duplicates()
df = df.merge(admit_times, on='admissionid', how='left')
df['dischargedat'] = pd.to_datetime(df['dischargedat'], unit='ms', errors='coerce')
df['LOS_remaining_days'] = (df['dischargedat'] - window_end).dt.total_seconds() / (24 * 3600)
df.drop(columns=['dischargedat'], inplace=True)

# Next-step RRT: is RRT active in the next window?
if rrt_raw is not None:
    rrt_next = rrt_raw.copy()
    rrt_next['rrt_next_timestep'] = rrt_next.groupby('admissionid')['rrt'].shift(-1).fillna(0).astype(int)
    df = df.merge(rrt_next[['admissionid', 'window', 'rrt_next_timestep']], on=['admissionid', 'window'], how='left')
    df['rrt_next_timestep'] = df['rrt_next_timestep'].fillna(0).astype(int)

# Next-step MV: is PEEP > 0 in the next window?
if 'peep' in df.columns:
    df['mv_active'] = np.where((df['peep'].notna()) & (df['peep'] > 0), 1, 0)
    df['mv_next_timestep'] = df.groupby('admissionid')['mv_active'].shift(-1).fillna(0).astype(int)
    df.drop(columns=['mv_active'], inplace=True)

# Save 

df.to_csv(output_path, index=False)
print(f'\nSaved {len(df)} rows ({df.admissionid.nunique()} admissions) to {output_path}')
print(f'Final columns: {list(df.columns)}')
print(f'\nLabel distributions:')
print(f'  90-day mortality positive rate: {df["90day_mortality"].mean():.3f}')
if 'rrt_next_timestep' in df.columns:
    print(f'  Next-step RRT positive rate: {df["rrt_next_timestep"].mean():.3f}')
if 'mv_next_timestep' in df.columns:
    print(f'  Next-step MV positive rate: {df["mv_next_timestep"].mean():.3f}')
print(f'  Remaining LOS (days) median: {df["LOS_remaining_days"].median():.1f}')
