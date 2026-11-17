# Foundation Model Encoders for Prediction-Guided Sepsis Phenotyping: Extending Temporal Representation Learning on AmsterdamUMCdb

MSc Artificial Intelligence capstone project — University of Galway.

This project investigates whether replacing the LSTM encoder in Hilders et al.'s (2026) prediction-guided sepsis phenotyping framework with a frozen pretrained time-series foundation model (MOMENT-1-base) produces better or different sepsis sub-phenotypes. The primary dataset used is AmsterdamUMCdb v1.0.2.

## Repository Structure

- `src/` — All notebooks for data extraction, preprocessing, model training, and clustering
- `data/` — Empty in this repo; data splits and cohort files are kept locally due to patient data privacy requirements
- `AmsterdamUMCdb/` — AmsterdamUMCdb Python package for data access
- `SepsisPhenotyping/` — Public reference implementation of Hilders et al. (2026), the anchor paper for this project

## Data

Data is sourced from AmsterdamUMCdb v1.0.2. Access requires registration and approval at [amsterdamumcdb.nl](https://amsterdamumcdb.nl). Raw data and processed splits are not included in this repository.

## Supervised by
Prof. Michael Madden, University of Galway
