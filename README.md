# Baseline Predictive Pipeline -- ETAI

**Beatriz Marques** · 20231605 · forked from [sofiacper/ETAI-Pipeline](https://github.com/sofiacper/ETAI-Pipeline)

Predictive pipeline for two-year recidivism using ProPublica's COMPAS dataset -- the data behind a 2016 investigation into a risk-assessment algorithm used by US courts. See `data/README.md` for the problem description and data dictionary.

The pipeline started from the course baseline and is improved each week; the progress table below logs the changes. Results: [`result_analysis.md`](result_analysis.md).

## Project structure

```
├── main.py                   # entry point: train, evaluate and save the model
├── predict.py                # entry point: predict a new CSV with the saved model (week 10)
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── data_diagnostics.py   # missingness-mechanism test, domain-rule checks, duplicate check
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py              # model construction (incl. the oblique random forest and AdaBoost builders, week 6; XGBoost, week 8) + VoteSelector, the feature-selection step (week 8) + build_pipeline(): preprocessing + selection + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6), several metrics and out-of-fold probabilities (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   ├── results.py            # saves each run's report to disk, and its scores to the runs table (week 7)
│   └── course_helpers.py     # last week's pipeline in one file, so notebook 08 runs on its own (copied from the course)
├── notebooks/
│   ├── 01_eda_introduction.ipynb   # EDA: missingness, invalid values, duplicates, correlation + VIF
│   ├── 02_preprocessing.ipynb      # encoder/scaler grid + paired 
│   ├── 03_cross_validation.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_ensembles.ipynb
│   ├── 06_boosting.ipynb
│   ├── 07_metrics.ipynb
│   └── 08_feature_selection.ipynb   # XGBoost, feature selection (VoteSelector), the leakage trap, trees beyond yes/no
├── results/                  # created automatically -- one report per run + runs.csv, the table of all runs (not tracked in git)
├── models/                   # created automatically -- model.joblib, the final fitted pipeline (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── compas_new_people.csv # 20 people without the target, to try predict.py
    ├── diagnosis_log.json    # diagnosis log
    └── README.md             # problem description + full data dictionary
```

## Pipeline progress

Updated after each practical class; a running log, not a fixed syllabus.

| Week | Focus | Added to the pipeline |
|------|-------|-----------------------|
| 2 | Introduction & baseline pipeline | Project structure; single train/test split; minimal preprocessing (drop rows with missing values, one-hot encode); logistic regression baseline; fairness check comparing our model's and COMPAS's false-positive rate by race; each run's report saved to `results/`. |
| 3 | EDA + preprocessing | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid values, duplicate check). `src/preprocessing.py`: leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, the train/test split. Target encoding + standard scaling chosen by a grid over 15 repeated splits; three redundant columns dropped (correlation + VIF). `config.yaml` gains `diagnostics` and `preprocessing`. |
| 4 | Preprocessing inside the pipeline + cross-validation | **Locked final test set** (20%, stratified, seed 42) set aside by `split_dev_test()` and never scored; models judged by **stratified 5-fold CV** of the whole pipeline on the development set (mean ± std, train-validation gap); out-of-fold predictions feed the classification report and fairness check. Target encoding switched to scikit-learn's cross-fitting `TargetEncoder`; encoder/scaler set in `config.yaml` (target + robust). **Fixes** in `clean_dataset()`: real `NaN`s in categorical columns became the string `"nan"`, so 229 `c_charge_degree` gaps were never imputed or flagged (fixed by adding `"nan"` to `diagnostics.placeholder_tokens`); it no longer drops rows (de-duplication moved to a training-only `drop_duplicate_rows()` before the split, so the same cleaning runs on new data). `src/data_diagnostics.py` removed: `flag_invalid_values` moved into `preprocessing.py`, EDA-only checks live in the notebooks. `dummy` floor and untuned `random_forest` added; final model refit on the development set; `config.yaml` gains `test_set` and `cv`. See [`Week 4 — Preprocessing inside the pipeline and cross-validation`](result_analysis.md). |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: **Optuna** (seeded TPE) on the development set, each candidate scored by stratified 5-fold CV of the **whole** pipeline on the same folds; **nested CV** estimates the tuning procedure honestly and feeds the reports; the "best trial" score is shown as an optimism check; the final model is the tuned pipeline refit on all development rows. `build_pipeline()` in `src/model.py` builds the preprocessing + model `Pipeline`. `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model); model becomes the decision tree; `optuna` added to `requirements.txt`. See [`Week 5 - Hyperparameter tuning`](result_analysis.md). |
| 6 | Ensembles: random forest, oblique random forest, AdaBoost | New model types `oblique_random_forest` (`SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` in `src/model.py`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); search spaces for the three ensembles in `config.yaml`. `corrected_paired_test()` in `src/evaluate.py` (Nadeau & Bengio) calls a "winner" only when the difference exceeds fold-to-fold noise; `scipy` listed in `requirements.txt`. See [`Week 6 - Ensembles`](result_analysis.md). |
| 7 | Gradient boosting + evaluating with probabilities | New model types `gradient_boosting`, `hist_gradient_boosting`, `naive_bayes`; gradient boosting is now the model `main.py` runs. Evaluation reports **several metrics** (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold **probabilities** feed the reports, with a confusion matrix. New `save_scores()` in `src/results.py`: every run adds a row to `results/runs.csv` (model, settings, per-fold scores, no-model floor, untuned score), so runs on the same data and folds can be compared with `corrected_paired_test()`. See "Model evaluation" below and [`Week 7 — Boosting and metric evaluation`](result_analysis.md). |
| 8 | XGBoost + feature selection | New model type `xgboost` (`XGBClassifier`), now the model `main.py` runs. **Feature selection** as a learned Pipeline step: `VoteSelector` in `src/model.py` (random forest, gradient boosting and L1 logistic regression vote; features with at least `min_votes` votes are kept), placed between `"prep"` and `"model"` when `feature_selection` is enabled in `config.yaml`, so it is re-fitted on the training rows of every fold; with tuning on, `select__min_votes` is tuned and assessed by nested CV. The pipeline caches fitted preprocessing and selection steps during a run (`Pipeline(memory=...)`, a temporary folder deleted at the end). The kept features are printed and the scoreboard records the selection setting. See "Feature selection" below and [`Week 8 — XGBoost and feature selection`](result_analysis.md). |
| 10 | Missing values + a deployable pipeline | The numeric imputer is a config choice (`preprocessing.imputation.numeric_strategy`: `median`, `mean`, `knn`, `iterative` or `random_forest`) via `_NUMERIC_IMPUTERS` in `src/preprocessing.py`; numeric columns are scaled **before** imputation; the runs table and report record the imputer. `main.py` saves the final pipeline to `models/model.joblib` (`output.model_path`, git-ignored); new `predict.py` writes id, probability and prediction for each row of a new CSV (`data/compas_new_people.csv` to try it); `joblib` added to `requirements.txt`; duplicate `data/compas_two_year_recidivism (1).csv` removed. See "Missing values" and "Predicting new data" below. |

## Preprocessing decisions

Summary of the diagnosis in `01_eda_introduction.ipynb` and the empirical grid in `02_preprocessing.ipynb`.

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (not a model feature) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | out-of-range or negative values | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact duplicates sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant (r = 1.00, or an exact sum caught by VIF for `juvenile_total`) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.

## Hyperparameter tuning

1. **Nested cross-validation** -- for each of the 5 outer CV folds, runs a full Optuna study on the outer-training rows only (each trial = 5-fold CV of the whole pipeline), refits the best candidate on those rows and scores it on the outer-validation fold. This is the honest estimate of the tuned model.
2. **Tuning on the whole development set** -- the same study, once, on every development row: these are the hyperparameters the final model uses.

The best trial's score is printed too, but it is **optimistic** (the maximum of many noisy CV scores) -- report the nested one. Search spaces live in `config.yaml` under `tuning.search_spaces`, one per model type, keyed by Pipeline parameter name (`model__max_depth`); a model without a search space can't be tuned until you add one. Cost: `(cv.n_splits + 1) x n_trials x tuning.n_splits` fits.

## Model evaluation

**Metrics.** `cv.metrics` lists what every run reports; `cv.selection_metric` is the one that matters: tuning maximises it and the scoreboard ranks runs by it. The default is **ROC-AUC**: it scores how well the model *ranks* people by risk, without choosing a threshold, so COMPAS's own 1-10 risk score can be judged the same way. Accuracy, precision, recall and F1 depend on the threshold that turns a probability into a yes/no (0.5 in this pipeline). `fbeta_<beta>` is F-beta: `fbeta_2` counts recall twice as much as precision.

**The scoreboard.** Every `python main.py` adds one row to `results/runs.csv` (`save_scores()` in `src/results.py`): the run id, the data file, the model, whether it was tuned, the encoder and scaler, the model's params (and the tuned ones), the selection metric with its mean and std, the **untuned** score of the same model, the **no-model floor** (a dummy on the same folds) and the score on each fold (`fold_1` ... `fold_5`, from nested CV when tuning is on). The console prints the runs on the same data and metric, best first. The full report of each run is in `results/run_<run id>.txt`.

Two runs can only be compared if they used the **same data, cleaning and folds**: when you change the data file, the cleaning rules or `cv.random_state`, delete `results/runs.csv` and start a new table. (You could make this more automatic.) To compare two runs, from the project root:

```python
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
corrected_paired_test(runs.loc[0, folds], runs.loc[1, folds], test_train_ratio=1 / 4)   # rows 0 and 1; 5-fold: 1 / (5 - 1)
```

It returns the mean difference, the folds the first run wins, t and the p-value.

## Feature selection

Set `feature_selection.enabled` in `config.yaml` to add the `"select"` step between preprocessing and the model. `VoteSelector` fits three models on the training rows (a random forest, gradient boosting, and an L1-penalised logistic regression); each votes for the features it finds useful (above-average importance, or a non-zero coefficient), and features with at least `min_votes` votes are kept. Because it learns from the labels, it lives **inside** the Pipeline and is re-fitted in every fold; selecting features on all rows before cross-validation would make the score optimistic. With tuning on, `feature_selection.search_space` is added to the model's search space, so `min_votes` is chosen with the model's hyperparameters and assessed by nested CV. The features the final model keeps are printed at the end of each run.

**XGBoost** is a separate package: see "Installing XGBoost" below if `import xgboost` fails.

## Missing values

I ran XGBoost (feature selection on, tuning off) with four numeric imputers and the same folds: median 0.721 ± 0.013 ROC-AUC, KNN 0.720 ± 0.012, iterative (MICE) 0.721 ± 0.014 and random forest 0.721 ± 0.013. The corrected paired test against median gave p = 0.86 (KNN), 0.95 (iterative) and 0.82 (random forest), so none is better than median, and the differences (at most 0.0003 AUC) are far smaller than the fold-to-fold noise. I keep `median`: it is the simplest and fastest, and the model-based imputers cost more time for no gain.

## Predicting new data

`main.py` saves the final model to `models/model.joblib`. To predict a new CSV:
```bash
python predict.py data/compas_new_people.csv predictions.csv
```
The output has 20 rows, one per person, with three columns: `id`, `probability` (of two-year recidivism) and `prediction` (1 if the probability is at least 0.5, else 0). In my run 6 of the 20 were predicted 1. `models/` and `predictions.csv` are git-ignored.

## Setup

Run once per machine:

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once, or use `venv\Scripts\activate.bat` in cmd.exe.

## Installing XGBoost

`xgboost` is in `requirements.txt`, so `pip install -r requirements.txt` (venv active) installs it. On its own: `pip install xgboost`. Check it worked, with the venv active:
```bash
python -c "import xgboost; print(xgboost.__version__)"
```

**Windows:** nothing extra. If the install fails, update pip first (`python -m pip install --upgrade pip`) and make sure your Python is 64-bit (`python -c "import struct; print(struct.calcsize('P') * 8)"` should print 64).

**macOS:** XGBoost needs the **OpenMP** library (for running on several CPU cores), which macOS doesn't ship. If `import xgboost` fails with `XGBoostError`, `Library not loaded: libomp.dylib` or similar:
1. install [Homebrew](https://brew.sh) if you don't have it (the command is on its home page);
2. run `brew install libomp`;
3. open a **new** terminal, activate the venv, and try the import again.

Works on Apple Silicon and Intel Macs. With conda, `conda install -c conda-forge xgboost` also installs OpenMP.

**Linux:** on most distributions `pip install xgboost` just works. Two things can go wrong:
- it downloads a very large extra package (`nvidia-nccl-...`, a few hundred MB) for NVIDIA GPUs. If you have no NVIDIA GPU, or the download is too slow, install the CPU-only version instead: `pip install xgboost-cpu` (the code is identical: still `import xgboost`). If you do this, replace `xgboost` with `xgboost-cpu` in your copy of `requirements.txt`, so a reinstall doesn't fetch the large package again.
- a very old distribution may have no ready-made version and try to compile it. Update pip first (`python -m pip install --upgrade pip`); recent releases need glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, Fedora 29+).

**In a notebook (Jupyter / VS Code): `No module named 'xgboost'`, although you installed it.** The notebook runs on a different Python from the one `pip` installed into (e.g. you installed into the venv, but the notebook uses the system Python). Either:
- install from inside the notebook, into the Python it actually uses: run `%pip install xgboost` in a cell, then **restart the kernel**; or
- make your venv available as a kernel, once, with the venv active:
  ```bash
  pip install ipykernel
  python -m ipykernel install --user --name etai --display-name "ETAI (venv)"
  ```
  then choose **ETAI (venv)** as the notebook's kernel. The notebook then sees every package in `requirements.txt`.

To check which Python a notebook is using: `import sys; print(sys.executable)`. In the terminal: `which python` (macOS/Linux) or `where python` (Windows). They should point into the same `venv` folder.

## Running the pipeline

With the environment active, from the project root:

```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above; minutes, not seconds, for the ensembles), and prints, in order:
- the model, the selection metric and the CV design; the size of the development set and of the locked test set (which is **not** used)
- **Default hyperparameters**: the cross-validation score on the selection metric, the train-validation gap (how much better the model looks on rows it trained on: a large gap means it memorised) and the no-model floor
- with tuning on, **Tuning**: the nested cross-validation score (the honest score of the tuned model) and the chosen hyperparameters
- **All metrics**: mean ± std of every metric
- **Scoreboard**: this run next to the earlier runs on the same data and metric, best first
- the final model (refit on all development rows), the features it keeps (with feature selection on), where the full report was saved and where the model was saved for `predict.py`

The full detail -- per-fold tables for every metric, the tuning report, the out-of-fold confusion matrix and classification report, the false-positive-rate-by-race comparison with COMPAS's own score -- goes to the results file.

All of this is also saved to a timestamped file in `results/` (e.g. `results/run_20260916_143012.txt`). `results/` is created automatically and is not tracked in git (see `.gitignore`).

## Push to GitHub via Terminal

From the project root, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

If `git push` rejects your GitHub password, HTTPS needs a **Personal Access Token (PAT)**:
1. **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. Paste the token as the password when `git push` prompts (username stays the same).
3. To avoid being asked every time: `git config --global credential.helper manager` (Windows) or `store` (plaintext cache; personal machines only).

Alternative: add an SSH key (`ssh-keygen -t ed25519`, public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL.

## Result Analysis

The full weekly analysis is in [`result_analysis.md`](result_analysis.md).
