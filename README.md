# Baseline Predictive Pipeline -- ETAI

**Beatriz Marques** · 20231605 · forked from [sofiacper/ETAI-Pipeline](https://github.com/sofiacper/ETAI-Pipeline)

Predictive pipeline for two-year recidivism using ProPublica's COMPAS dataset -- the data behind a 2016 investigation into a risk-assessment algorithm used by US courts to inform bail and sentencing decisions. See `data/README.md` for the full problem description and data dictionary.

The pipeline started from the course baseline and is improved each week. Changes are logged in the progress table below.

## Project structure

```
├── main.py                   # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── data_diagnostics.py   # missingness-mechanism test, domain-rule checks, duplicate check
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py              # model construction (incl. the oblique random forest and AdaBoost builders, week 6) + build_pipeline(): preprocessing + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6), several metrics and out-of-fold probabilities (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk, and its scores to the runs table (week 7)
├── notebooks/
│   ├── 01_eda_introduction.ipynb   # EDA: missingness, invalid values, duplicates, correlation + VIF
│   ├── 02_preprocessing.ipynb      # encoder/scaler grid + paired 
│   ├── 03_cross_validation.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_ensembles.ipynb
│   ├── 06_boosting.ipynb
│   └── 07_metrics.ipynb
├── results/                  # created automatically -- one report per run + runs.csv, the table of all runs (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── diagnosis_log.json    # diagnosis log
    └── README.md             # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Focus | Added to the pipeline |
|------|-------|-----------------------|
| 2 | Introduction & baseline pipeline | Project structure; single train/test split (no cross-validation); minimal preprocessing (drop rows with missing values, one-hot encode categoricals); logistic regression baseline; simple fairness check comparing our model's and COMPAS's false-positive rate by race; train-vs-test accuracy reporting; each run's report saved to `results/`. |
| 3 | EDA + preprocessing | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, duplicate check). `src/preprocessing.py` now handles leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, and the train/test split, replacing the old `dropna()`/`pd.get_dummies()`. Encoder/scaler (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits. Three redundant columns dropped (correlation + VIF). `config.yaml` gains `diagnostics` and `preprocessing` sections. |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `\"nan\"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `\"nan\"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see [`Week 4 — Preprocessing inside the pipeline and cross-validation`](result_analysis.md).|
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported \"best trial\" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see [`Week 5 - Hyperparameter tuning`](result_analysis.md).|
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a \"winner\" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see [`Week 6 - Ensembles`](result_analysis.md).|
| 7 | Gradient boosting + evaluating with probabilities | Three new model types in `src/model.py`: `gradient_boosting` (scikit-learn's `GradientBoostingClassifier`: each shallow tree fits the gradient of the log-loss left by the trees before it), `hist_gradient_boosting` (the same idea on binned features: much faster on large data, and it handles missing values itself) and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline); gradient boosting is now the model `main.py` runs, each with its search space in `config.yaml`; evaluation now reports **several metrics** (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold **probabilities** (not just labels) feed the reports, with a confusion matrix; `class_weight: balanced` in the commented examples of the models that support it; new `save_scores()` in `src/results.py`: every run adds a row to `results/runs.csv` (the model and its settings, the score on each fold, the no-model floor and the untuned score on the same folds), so any two runs on the same data and folds can be compared with `corrected_paired_test()`; the printed output was reorganised to be easier to read, with the full tables in the results file (optional: keep your own format if you prefer) -- see "Model evaluation" below and [`Week 7 — Boosting and metric evaluation`](result_analysis.md).|

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
- the final model (refit on all development rows) and where the full report was saved

The full detail -- per-fold tables for every metric, the tuning report, the out-of-fold confusion matrix and classification report, the false-positive-rate-by-race comparison with COMPAS's own score -- goes to the results file.

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Result Analysis

The full weekly analysis is in [`result_analysis.md`](result_analysis.md).
