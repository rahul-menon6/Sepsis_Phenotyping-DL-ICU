"""
Extracts the Sepsis-3 cohort from AmsterdamUMCdb following Hilders et al. (2026).
Inclusion criteria:
    - Adult ICU admissions (agegroup documented, location contains 'IC')
    - Sepsis-3: suspected infection + SOFA >= 2 within first 24h of ICU admission

Output: data/cohort/sepsis3_cohort.csv
"""

import os
import configparser
import psycopg2
import sqlalchemy
import pandas as pd
import amsterdamumcdb as adb

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'AmsterdamUMCdb', 'config.ini')
sample_path = os.path.join(os.path.dirname(__file__), '..', 'AmsterdamUMCdb', 'config.SAMPLE.ini')
config.read(config_path if os.path.isfile(config_path) else sample_path)

output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cohort')
os.makedirs(output_dir, exist_ok=True)

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

engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{config['psycopg2']['username']}:{config['psycopg2']['password']}"
    f"@{config['psycopg2']['host']}:{config['psycopg2']['port']}/{config['psycopg2']['database']}"
    f"?options=-csearch_path%3Dpublic%20-cclient_encoding%3DWIN1252"
)
con = engine.connect()

print('Loading admissions...')
admissions = pd.read_sql("SELECT * FROM admissions", con)
print(f'  Total: {admissions.admissionid.nunique()}')

icu_admissions = admissions[admissions['location'].str.contains('IC', na=False)]
print(f'  ICU: {icu_admissions.admissionid.nunique()}')

adult_admissions = icu_admissions[
    icu_admissions['agegroup'].notna() &
    ~icu_admissions['agegroup'].isin(['None', ''])
]
print(f'  Adult ICU: {adult_admissions.admissionid.nunique()}')

print('Extracting Sepsis-3 cohort...')
sepsis_cohort = adb.get_sepsis_patients(con)
print(f'  Sepsis-3 (all): {sepsis_cohort.admissionid.nunique()}')

adult_icu_ids = set(adult_admissions['admissionid'])
sepsis_cohort = sepsis_cohort[sepsis_cohort['admissionid'].isin(adult_icu_ids)]
print(f'  Sepsis-3 adult ICU: {sepsis_cohort.admissionid.nunique()}')

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

output_path = os.path.join(output_dir, 'sepsis3_cohort.csv')
sepsis_cohort.to_csv(output_path, index=False)
print(f'\nSaved {sepsis_cohort.admissionid.nunique()} admissions to {output_path}')

pg_con.close()
