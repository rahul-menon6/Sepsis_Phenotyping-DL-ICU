"""
Extracts 4-hour windowed features for the Sepsis-3 cohort from AmsterdamUMCdb,
following Hilders et al. (2026). Produces one row per admission per 4-hour window
with 39 features matching Hilders' input schema.

Output: data/features/features_raw.csv

Usage:
    conda activate ml-env
    python src/02_extract_features.py
"""

import os
import configparser
import psycopg2
import sqlalchemy
import pandas as pd
import numpy as np

# Setup

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'AmsterdamUMCdb', 'config.ini')
sample_path = os.path.join(os.path.dirname(__file__), '..', 'AmsterdamUMCdb', 'config.SAMPLE.ini')
config.read(config_path if os.path.isfile(config_path) else sample_path)

cohort_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cohort', 'sepsis3_cohort.csv')
output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data', 'features')
output_path = os.path.join(output_dir, 'features_raw.csv')
os.makedirs(output_dir, exist_ok=True)

engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{config['psycopg2']['username']}:{config['psycopg2']['password']}"
    f"@{config['psycopg2']['host']}:{config['psycopg2']['port']}/{config['psycopg2']['database']}"
    f"?options=-csearch_path%3Dpublic%20-cclient_encoding%3DWIN1252"
)
con = engine.connect()

print('Loading cohort...')
cohort = pd.read_csv(cohort_path)
admission_ids = tuple(cohort['admissionid'].astype(int).tolist())
print(f'  {len(admission_ids)} admissions')

WINDOW_MINUTES = 240  # 4-hour windows

def assign_window(time_minutes):
    """Assign a measurement to a 4-hour window index (0-indexed from admission)."""
    return (time_minutes // WINDOW_MINUTES).astype(int)

def query_numericitems(itemids: list, label: str) -> pd.DataFrame:
    """Query numericitems for given itemids, filtered to cohort, returning time in minutes from admission."""
    ids = ', '.join(str(i) for i in itemids)
    sql = f"""
        SELECT
            n.admissionid,
            n.value,
            (n.measuredat - a.admittedat) / (1000.0 * 60) AS time_minutes
        FROM numericitems n
        LEFT JOIN admissions a ON n.admissionid = a.admissionid
        WHERE n.itemid IN ({ids})
        AND n.admissionid IN {admission_ids}
        AND (n.measuredat - a.admittedat) >= -(1000*60*30)
    """
    df = pd.read_sql(sqlalchemy.text(sql), con)
    df['feature'] = label
    return df

def query_listitems_gcs(component: str, eye_ids: list, motor_ids: list, verbal_ids: list) -> pd.DataFrame:
    """
    Query GCS components from listitems. Uses valueid mappings from gcs.sql.
    Returns numeric scores per admission per measurement time.
    """
    eye_ids_str    = ', '.join(str(i) for i in eye_ids)
    motor_ids_str  = ', '.join(str(i) for i in motor_ids)
    verbal_ids_str = ', '.join(str(i) for i in verbal_ids)

    sql = f"""
        SELECT
            eyes.admissionid,
            (eyes.measuredat - a.admittedat) / (1000.0 * 60) AS time_minutes,
            CASE eyes.itemid
                WHEN 6732  THEN 5 - eyes.valueid
                WHEN 13077 THEN eyes.valueid
                WHEN 14470 THEN eyes.valueid - 4
                WHEN 16628 THEN eyes.valueid - 4
                WHEN 19635 THEN eyes.valueid - 4
                WHEN 19638 THEN eyes.valueid - 8
            END AS eyes_score,
            CASE motor.itemid
                WHEN 6734  THEN 7 - motor.valueid
                WHEN 13072 THEN motor.valueid
                WHEN 14476 THEN motor.valueid - 6
                WHEN 16634 THEN motor.valueid - 6
                WHEN 19636 THEN motor.valueid - 6
                WHEN 19639 THEN motor.valueid - 12
            END AS motor_score,
            CASE verbal.itemid
                WHEN 6735  THEN 6 - verbal.valueid
                WHEN 13066 THEN verbal.valueid
                WHEN 14482 THEN verbal.valueid - 5
                WHEN 16640 THEN verbal.valueid - 5
                WHEN 19637 THEN verbal.valueid - 9
                WHEN 19640 THEN verbal.valueid - 15
            END AS verbal_score
        FROM listitems eyes
        LEFT JOIN admissions a ON eyes.admissionid = a.admissionid
        LEFT JOIN listitems motor ON
            eyes.admissionid = motor.admissionid AND
            eyes.measuredat  = motor.measuredat AND
            motor.itemid IN ({motor_ids_str})
        LEFT JOIN listitems verbal ON
            eyes.admissionid = verbal.admissionid AND
            eyes.measuredat  = verbal.measuredat AND
            verbal.itemid IN ({verbal_ids_str})
        WHERE eyes.itemid IN ({eye_ids_str})
        AND eyes.admissionid IN {admission_ids}
        AND (eyes.measuredat - a.admittedat) >= 0
    """
    return pd.read_sql(sqlalchemy.text(sql), con)

def query_vasopressor_rate() -> pd.DataFrame:
    """
    Query norepinephrine-equivalent vasopressor rate from drugitems.
    Returns total norepinephrine-equivalent dose per 4h window.
    Itemids: 7179 dopamine, 7178 dobutamine, 6818 epinephrine, 7229 norepinephrine.
    Conversion to norepinephrine equivalents follows standard clinical practice.
    """
    sql = f"""
        SELECT
            d.admissionid,
            d.itemid,
            (d.start - a.admittedat) / (1000.0 * 60) AS start_minutes,
            (d.stop  - a.admittedat) / (1000.0 * 60) AS stop_minutes,
            d.dose,
            d.doseunit,
            d.doserateperkg,
            d.doseunitid,
            d.doserateunitid,
            CASE a.weightgroup
                WHEN '59'  THEN 55
                WHEN '60'  THEN 65
                WHEN '70'  THEN 75
                WHEN '80'  THEN 85
                WHEN '90'  THEN 95
                WHEN '100' THEN 105
                WHEN '110' THEN 115
                ELSE 80
            END AS patientweight
        FROM drugitems d
        LEFT JOIN admissions a ON d.admissionid = a.admissionid
        WHERE d.ordercategoryid = 65
        AND d.itemid IN (7179, 7178, 6818, 7229)
        AND d.rate > 0.1
        AND d.admissionid IN {admission_ids}
    """
    return pd.read_sql(sqlalchemy.text(sql), con)

def query_iv_fluids() -> pd.DataFrame:
    """
    Query total IV fluid volume from drugitems per admission.
    Covers crystalloids and colloids administered as continuous infusions.
    """
    sql = f"""
        SELECT
            d.admissionid,
            (d.start - a.admittedat) / (1000.0 * 60) AS start_minutes,
            (d.stop  - a.admittedat) / (1000.0 * 60) AS stop_minutes,
            d.fluidin
        FROM drugitems d
        LEFT JOIN admissions a ON d.admissionid = a.admissionid
        WHERE d.ordercategoryid IN (
            55,  -- Infuus - Crystalloid
            17,  -- Infuus - Colloid
            114  -- Voeding Enteraal
        )
        AND d.fluidin > 0
        AND d.admissionid IN {admission_ids}
    """
    return pd.read_sql(sqlalchemy.text(sql), con)

def query_rrt() -> pd.DataFrame:
    """Query RRT (CRRT) sessions from listitems."""
    sql = f"""
        SELECT
            l.admissionid,
            (l.measuredat - a.admittedat) / (1000.0 * 60) AS time_minutes,
            1 AS rrt_active
        FROM listitems l
        LEFT JOIN admissions a ON l.admissionid = a.admissionid
        WHERE l.itemid = 14846
        AND l.admissionid IN {admission_ids}
        AND (l.measuredat - a.admittedat) >= 0
    """
    return pd.read_sql(sqlalchemy.text(sql), con)

def query_peep() -> pd.DataFrame:
    """Query PEEP settings from numericitems."""
    sql = f"""
        SELECT
            n.admissionid,
            n.value,
            (n.measuredat - a.admittedat) / (1000.0 * 60) AS time_minutes
        FROM numericitems n
        LEFT JOIN admissions a ON n.admissionid = a.admissionid
        WHERE n.itemid IN (
            12285, -- PEEP (Servo-i)
            12286  -- PEEP (Servo-u)
        )
        AND n.admissionid IN {admission_ids}
        AND (n.measuredat - a.admittedat) >= -(1000*60*30)
    """
    df = pd.read_sql(sqlalchemy.text(sql), con)
    df['feature'] = 'peep'
    return df

# Feature itemid mappings 

NUMERIC_FEATURES = {
    'heart_rate':    [6640],
    'spo2':          [6709],
    'resp_rate':     [8873, 12266, 8874],
    'sysbp':         [6641, 6678, 8841],
    'meanbp':        [6642, 6679, 8843],
    'diasbp':        [6643, 6680, 8842],
    'tempc':         [8658, 8659, 8662, 13058, 13059, 13060, 13061, 13062, 13063, 13952, 16110],
    'cvp':           [6655],
    'lactate':       [10053, 6837, 9580],
    'creatinine':    [6836, 9941, 14216],
    'bilirubin':     [6813, 9945],
    'platelet':      [6797, 9964, 10409, 14252],
    'hematocrit':    [6777, 11423, 11545],
    'wbc':           [6779, 9965],
    'ph':            [6848, 12310],
    'pao2':          [7433, 9996, 21214],
    'paco2':         [6846, 9990, 21213],
    'sodium':        [6840, 9555, 9924, 10284],
    'potassium':     [6835, 9556, 9927, 10285],
    'glucose':       [6833, 9557, 9947],
    'hemoglobin':    [6778, 9553, 9960, 10286],
    'bicarbonate':   [6810, 9992, 9993],
    'chloride':      [6819, 9558, 9930],
    'calcium':       [6806, 9930],
    'bun':           [6850, 9943],
    'ast':           [6806, 11990],
    'alt':           [6800, 11978],
    'inr':           [6822, 9950],
    'albumin':       [6801, 9975],
    'crp':           [6825, 10079],
    'inr':           [6789],
}

# Query all numeric features 

print('Querying numeric features...')
numeric_dfs = []
for feature, itemids in NUMERIC_FEATURES.items():
    print(f'  {feature}')
    df = query_numericitems(itemids, feature)
    numeric_dfs.append(df)

numeric_raw = pd.concat(numeric_dfs, ignore_index=True)
numeric_raw['window'] = assign_window(numeric_raw['time_minutes'])

# Aggregate numeric features per window 

print('Aggregating numeric features...')
numeric_agg = (
    numeric_raw[numeric_raw['value'].notna()]
    .groupby(['admissionid', 'window', 'feature'])['value']
    .mean()
    .reset_index()
    .pivot_table(index=['admissionid', 'window'], columns='feature', values='value')
    .reset_index()
)
numeric_agg.columns.name = None

# Query and aggregate PEEP  

print('Querying PEEP...')
peep_raw = query_peep()
peep_raw['window'] = assign_window(peep_raw['time_minutes'])
peep_agg = (
    peep_raw[peep_raw['value'].notna()]
    .groupby(['admissionid', 'window'])['value']
    .max()
    .reset_index()
    .rename(columns={'value': 'peep'})
)

# Query and aggregate GCS (min per window) 

print('Querying GCS...')
gcs_raw = query_listitems_gcs(
    component='gcs',
    eye_ids   =[6732, 13077, 14470, 16628, 19635, 19638],
    motor_ids =[6734, 13072, 14476, 16634, 19636, 19639],
    verbal_ids=[6735, 13066, 14482, 16640, 19637, 19640]
)
gcs_raw['window'] = assign_window(gcs_raw['time_minutes'])
gcs_agg = (
    gcs_raw
    .groupby(['admissionid', 'window'])
    .agg(gcs_eye=('eyes_score', 'min'), gcs_motor=('motor_score', 'min'), gcs_verbal=('verbal_score', 'min'))
    .reset_index()
)

# Query and aggregate vasopressor rate (max per window) 

print('Querying vasopressor rates...')
vaso_raw = query_vasopressor_rate()
vaso_raw['doserateperkg'] = vaso_raw['doserateperkg'].apply(lambda x: 0 if str(x) in ("b'0'", '0', 'False') else 1)

# compute gamma (µg/kg/min) then convert to norepinephrine equivalents
def compute_gamma(row):
    w = row['patientweight']
    d, uid, ruid, perkg = row['dose'], row['doseunitid'], row['doserateunitid'], row['doserateperkg']
    if perkg == 0 and uid == 11 and ruid == 4:
        return d / w if w > 0 else d / 80
    elif perkg == 0 and uid == 10 and ruid == 4:
        return d * 1000 / w if w > 0 else d * 1000 / 80
    elif perkg == 0 and uid == 10 and ruid == 5:
        return d * 1000 / (w * 60) if w > 0 else d * 1000 / (80 * 60)
    elif perkg == 1 and uid == 11 and ruid == 4:
        return d
    elif perkg == 1 and uid == 11 and ruid == 5:
        return d / 60
    return np.nan

vaso_raw['gamma'] = vaso_raw.apply(compute_gamma, axis=1)

# norepinephrine equivalents conversion (standard clinical conversion factors)
NE_EQUIV = {7229: 1.0, 6818: 1.0, 7179: 0.01, 7178: 0.01}
vaso_raw['ne_equiv'] = vaso_raw['gamma'] * vaso_raw['itemid'].map(NE_EQUIV)

# assign to windows based on drug start time
vaso_raw['window'] = assign_window(vaso_raw['start_minutes'].clip(lower=0))
vaso_agg = (
    vaso_raw.groupby(['admissionid', 'window'])['ne_equiv']
    .max()
    .reset_index()
    .rename(columns={'ne_equiv': 'vaso_rate'})
)

# IV fluids- sum per window 

print('Querying IV fluids...')
fluid_raw = query_iv_fluids()
fluid_raw['window'] = assign_window(fluid_raw['start_minutes'].clip(lower=0))
fluid_agg = (
    fluid_raw[fluid_raw['fluidin'].notna()]
    .groupby(['admissionid', 'window'])['fluidin']
    .sum()
    .reset_index()
    .rename(columns={'fluidin': 'fluid_window'})
)

# RRT- max per window, binary

print('Querying RRT...')
rrt_raw = query_rrt()
rrt_raw['window'] = assign_window(rrt_raw['time_minutes'])
rrt_agg = (
    rrt_raw.groupby(['admissionid', 'window'])['rrt_active']
    .max()
    .reset_index()
    .rename(columns={'rrt_active': 'rrt'})
)

# admission-level time grid 

print('Building time grid...')
admissions_df = pd.read_sql(
    sqlalchemy.text(f"SELECT admissionid, admittedat, dischargedat FROM admissions WHERE admissionid IN {admission_ids}"),
    con
)
admissions_df['los_minutes'] = (admissions_df['dischargedat'] - admissions_df['admittedat']) / (1000 * 60)
admissions_df['n_windows'] = (admissions_df['los_minutes'] / WINDOW_MINUTES).fillna(1).apply(np.ceil).astype(int).clip(lower=1)

time_grid = pd.concat([
    pd.DataFrame({'admissionid': row['admissionid'], 'window': range(int(row['n_windows']))})
    for _, row in admissions_df.iterrows()
], ignore_index=True)

time_grid['start_time'] = admissions_df.set_index('admissionid').loc[
    time_grid['admissionid'], 'admittedat'
].values + time_grid['window'] * WINDOW_MINUTES * 60 * 1000

# Merge all features onto time grid 

print('Merging features...')
features = time_grid.copy()
for df, name in [
    (numeric_agg, 'numeric'),
    (peep_agg,    'peep'),
    (gcs_agg,     'gcs'),
    (vaso_agg,    'vaso'),
    (fluid_agg,   'fluid'),
    (rrt_agg,     'rrt'),
]:
    features = features.merge(df, on=['admissionid', 'window'], how='left')
    print(f'  merged {name}')

# Add static demographic features from admissions 

print('Adding demographics...')
demo_cols = [c for c in ['admissionid', 'gender', 'agegroup', 'weightgroup', 'heightgroup'] if c in cohort.columns]
demo = cohort[demo_cols].drop_duplicates()
features = features.merge(demo, on='admissionid', how='left')

# encode gender
features['gender'] = np.where(features['gender'] == 'Man', 1, 0)

# Save 

features['start_time'] = pd.to_datetime(features['start_time'], unit='ms')
print('\nMissingness per feature:')
feature_cols = [c for c in features.columns if c not in ['admissionid', 'window', 'start_time', 'agegroup', 'weightgroup', 'gender']]
print((features[feature_cols].isna().sum() / len(features)).round(2).sort_values(ascending=False).to_string())

features.to_csv(output_path, index=False)
print(f'\nSaved {len(features)} rows ({features.admissionid.nunique()} admissions) to {output_path}')
print(f'Columns: {list(features.columns)}')

con.close()
