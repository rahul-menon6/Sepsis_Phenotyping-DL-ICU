"""
01_extract_cohort.py

Extracts the Sepsis-3 cohort from AmsterdamUMCdb following Hilders et al. (2026):
    - Adult patients (age group not 'None' / agegroup documented)
    - ICU admissions only (location contains 'IC')
    - Sepsis-3 criteria met within 24h of ICU admission:
        * Suspected infection (via get_infection_patients)
        * SOFA total score >= 2 (via get_sofa_admission, first 24h)

Output:
    data/cohort/sepsis3_cohort.csv
    Contains admissionid, patientid, age/location metadata, and SOFA subscores.
    No raw clinical measurements — cohort IDs only.

Usage:
    conda activate ml-env
    python src/01_extract_cohort.py
"""

import os
import configparser
import psycopg2
import sqlalchemy
import pandas as pd
import amsterdamumcdb as adb

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'AmsterdamUMCdb', 'config.ini')
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'AmsterdamUMCdb', 'config.SAMPLE.ini')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'cohort')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'sepsis3_cohort.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

config = configparser.ConfigParser()
config.read(CONFIG_PATH if os.path.isfile(CONFIG_PATH) else SAMPLE_PATH)

# ── Database connection ───────────────────────────────────────────────────────

print('Connecting to PostgreSQL...')
pg_con = psycopg2.connect(
    database=config['psycopg2']['database'],
    user=config['psycopg2']['username'],
    password=config['psycopg2']['password'],
    host=config['psycopg2']['host'],
    port=config['psycopg2']['port']
)
pg_con.set_client_encoding('WIN1252')
pg_con.set_session(autocommit=True)

cursor = pg_con.cursor()
cursor.execute("SET SCHEMA 'public'")

# SQLAlchemy engine required by amsterdamumcdb package
engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{config['psycopg2']['username']}:{config['psycopg2']['password']}"
    f"@{config['psycopg2']['host']}:{config['psycopg2']['port']}/{config['psycopg2']['database']}"
    f"?options=-csearch_path%3Dpublic%20-cclient_encoding%3DWIN1252"
)
con = engine.connect()
con.execute(sqlalchemy.text("SET CLIENT_ENCODING TO 'WIN1252'"))

# ── Load admissions table ─────────────────────────────────────────────────────

print('Loading admissions...')
admissions = pd.read_sql("SELECT * FROM admissions", con)
print(f'  Total admissions: {admissions.admissionid.nunique()}')

# ── Filter: ICU only ──────────────────────────────────────────────────────────

# Hilders et al.: ICU admissions only (not MCU/step-down)
icu_admissions = admissions[admissions['location'].str.contains('IC', na=False)]
print(f'  ICU admissions: {icu_admissions.admissionid.nunique()}')

# ── Filter: Adults only ───────────────────────────────────────────────────────

# AmsterdamUMCdb agegroup field: '18-39', '40-49', ... '80+'; non-adults recorded differently
# Exclude agegroup that indicates paediatric/unknown
adult_admissions = icu_admissions[
    icu_admissions['agegroup'].notna() &
    ~icu_admissions['agegroup'].isin(['None', ''])
]
print(f'  Adult ICU admissions: {adult_admissions.admissionid.nunique()}')

# ── Sepsis-3 cohort extraction ────────────────────────────────────────────────

# get_sepsis_patients internally calls:
#   1. get_infection_patients (suspected infection from diagnoses/antibiotics/cultures)
#   2. get_sofa_admission (SOFA from first 24h, baseline assumed 0)
# and returns admissions where sofa_total_score >= 2
print('Extracting Sepsis-3 cohort...')
sepsis_cohort = adb.get_sepsis_patients(con)
print(f'  Sepsis-3 admissions (all): {sepsis_cohort.admissionid.nunique()}')

# ── Intersect with adult ICU admissions ───────────────────────────────────────

adult_icu_ids = set(adult_admissions['admissionid'])
sepsis_cohort = sepsis_cohort[sepsis_cohort['admissionid'].isin(adult_icu_ids)]
print(f'  Sepsis-3 adult ICU admissions: {sepsis_cohort.admissionid.nunique()}')

# ── Merge with admission metadata ─────────────────────────────────────────────

metadata_cols = [
    'admissionid', 'patientid', 'admittedat', 'dischargedat',
    'location', 'urgency', 'agegroup', 'gender', 'weightgroup', 'heightgroup'
]
available_cols = [c for c in metadata_cols if c in admissions.columns]
sepsis_cohort = pd.merge(
    sepsis_cohort,
    admissions[available_cols],
    on='admissionid',
    how='left',
    suffixes=('', '_adm')
)

# ── Save ──────────────────────────────────────────────────────────────────────

sepsis_cohort.to_csv(OUTPUT_PATH, index=False)
print(f'\nCohort saved to: {OUTPUT_PATH}')
print(f'Final cohort size: {sepsis_cohort.admissionid.nunique()} admissions')
print(f'Columns: {list(sepsis_cohort.columns)}')

# ── Summary ───────────────────────────────────────────────────────────────────

print('\nSOFA score distribution:')
print(sepsis_cohort['sofa_total_score'].describe())

pg_con.close()
