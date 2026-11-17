# Foundation Model Encoders for Prediction-Guided Sepsis Phenotyping

Extending temporal representation learning on AmsterdamUMCdb.

MSc Artificial Intelligence capstone project, University of Galway. Supervised by Prof. Michael Madden.

This project extends Hilders et al. (2026), who identify sepsis sub-phenotypes by clustering the embeddings of an LSTM trained to predict ICU outcomes. I replace the LSTM encoder with a frozen pretrained time-series foundation model (MOMENT-1-base) and compare the predictions and phenotypes that come out. Everything else in the framework stays the same.

## Results

Test-set performance, 4-task model (AmsterdamUMCdb, 2,644 Sepsis-3 admissions):

| Task | LSTM | MOMENT (frozen) |
|---|---|---|
| Mortality (AUROC) | 0.69 | 0.64 |
| RRT (Renal Replacament Therapy) (AUROC) | 0.75 | 0.75 |
| Length of stay (MSE) | 48 | 14.4 |

Mechanical Ventilation (MV) is left out of the table. About 98% of windows are positive in this cohort, so the AUROC isn't a meaningful comparison.

Clustering (window-level K-means on validation embeddings):

| | LSTM | MOMENT |
|---|---|---|
| Best k | 4 | 3 |
| Silhouette | 0.299 | 0.181 |

The Adjusted Rand Index between the two sets of clusters is 0.099, so the encoders pick up quite different structure.

The LSTM beats the frozen MOMENT encoder on mortality and on cluster quality. Two likely reasons: MOMENT is pretrained on non-clinical time series, and its patch design collapses 32 hours of data into one embedding point, which limits clustering resolution. LOS MSE favours MOMENT, but LOS is down-weighted in the loss, so that comparison is looser.

## Pipeline

PostgreSQL load → Sepsis-3 cohort → 4-hour windowed features → outlier removal → forward-fill → missingness filtering → median imputation → labels (90-day mortality, LOS, next-step RRT, next-step MV) → patient-level train/val/test split (1,850 / 397 / 397).

The final feature set has 33 clinical features. My cohort is larger than Hilders' (2,644 vs 1,535 admissions). I couldn't reproduce their exact cohort without their unpublished extraction code.

## Repository structure

```
src/
  01_extract_cohort.ipynb
  02_extract_features.ipynb
  03_preprocess.ipynb
  04_split_and_normalize.ipynb
  06_train_baseline_LSTM.ipynb
  07_train_baseline_moment.ipynb
  08_cluster_phenotypes.ipynb
data/        # empty, see below
```

Run the notebooks in numeric order from inside `src/`. After any change to preprocessing or the feature list, re-run 06, 07 and 08 in sequence.

## Data and privacy

This repo contains no patient data. Cohort files, train/val/test splits, cached embeddings and model checkpoints are kept locally, and notebook outputs are cleared.

The data comes from AmsterdamUMCdb v1.0.2, which needs registration and approval at [amsterdamumcdb.nl](https://amsterdamumcdb.nl/).

## Setup

- Python 3.11
- PyTorch, `momentfm`, `scikit-learn`, `pandas`
- [`amsterdamumcdb`](https://github.com/AmsterdamUMC/AmsterdamUMCdb) Python package
- PostgreSQL, with AmsterdamUMCdb loaded locally

## References

- Hilders et al. (2026), prediction-guided sepsis phenotyping. Public code: *[[github](https://github.com/paulhilders/SepsisPhenotyping)]*
- Goswami et al. (2024), MOMENT: A Family of Open Time-series Foundation Models
