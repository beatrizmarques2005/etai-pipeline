# Baseline Predictive Pipeline -- ETAI

**Beatriz Marques** · 20231605 · forked from [sofiacper/ETAI-Pipeline](https://github.com/sofiacper/ETAI-Pipeline)

Predictive pipeline for two-year recidivism using ProPublica's COMPAS dataset -- the data behind a 2016 investigation into a risk-assessment algorithm used by US courts to inform bail and sentencing decisions. See `data/README.md` for the full problem description and data dictionary.

The pipeline started from the course baseline and is improved each week. Changes are logged in the progress table below.

## Project structure

```
├── main.py                   # entry point: runs the whole pipeline
├── config.yaml               # all tunable settings
├── requirements.txt
├── src/
│   ├── data.py               # data loading
│   ├── data_diagnostics.py   # missingness-mechanism test, domain-rule checks, duplicate check
│   ├── preprocessing.py      # leak-safe cleaning, preprocessing pipeline, train/test split
│   ├── model.py              # model construction
│   ├── evaluate.py           # accuracy + fairness check
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report
├── notebooks/
│   ├── 01_eda_introduction.ipynb   # EDA: missingness, invalid values, duplicates, correlation + VIF
│   ├── 02_preprocessing.ipynb      # encoder/scaler grid + paired 
│   └── 03_cross_validation.ipynb      
├── results/                  # created automatically, one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── diagnosis_log.json    # diagnosis log
    └── README.md             # problem description + data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Focus | Added to the pipeline |
|------|-------|-----------------------|
| 2 | Introduction & baseline pipeline | Project structure; single train/test split (no cross-validation); minimal preprocessing (drop rows with missing values, one-hot encode categoricals); logistic regression baseline; simple fairness check comparing our model's and COMPAS's false-positive rate by race; train-vs-test accuracy reporting; each run's report saved to `results/`. |
| 3 | EDA + preprocessing | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, duplicate check). `src/preprocessing.py` now handles leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, and the train/test split, replacing the old `dropna()`/`pd.get_dummies()`. Encoder/scaler (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits. Three redundant columns dropped (correlation + VIF). `config.yaml` gains `diagnostics` and `preprocessing` sections. |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see [`Week 4 — Preprocessing inside the pipeline and cross-validation`](result_analysis.md).|
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see [`Week 5 - Hyperparameter tuning`](result_analysis.md).|
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see [`Week 6 - Ensembles`](result_analysis.md).|

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

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above), and prints:
- **a per-fold cross-validation table** -- train and validation accuracy for each of the 5 folds, the gap between them, and their mean ± std. Comparing train and validation is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises.
- with tuning on: the nested cross-validation table (the honest score of the tuned model, and the hyperparameters chosen in each fold), and the tuning report (best hyperparameters, top trials, and the optimism check)
- a classification report on the out-of-fold predictions (from nested CV when tuning is on)
- a false-positive-rate-by-race comparison between our model and COMPAS's own score (same rows)
- the final model -- the same pipeline (tuned, if tuning is on) refit on all development rows (CV estimated how good it is; this is the model itself)
- a reminder of how many rows are in the locked test set -- which is **not** evaluated

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

## Model evaluation

### Ensembles (Week 6)

All rows from `python main.py` with that `model.type` (accuracy, 5-fold CV on the development set; tuning: 30 Optuna trials, 5 inner folds; nested CV = 5 outer folds). The test set was not used.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | `max_iter=2000` |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.683 | `n_estimators=200`, `max_depth=22`, `min_samples_leaf=42`, `max_features=0.5` |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | `n_estimators=200`, `tree__max_depth=6`, `tree__min_samples_leaf=3`, `project__n_components=5` |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | `estimator__max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

Corrected paired t-test (Nadeau & Bengio, `test_train_ratio=1/4`), tuned AdaBoost vs tuned decision tree: +0.010, 4/5 folds, t = 1.68, p = 0.17; vs logistic regression: +0.010, 3/5 folds, t = 1.01, p = 0.37.

**Which model, and is its lead real?** AdaBoost has the best nested CV accuracy (0.682), but its lead is about 0.001 over the random forest, 0.003 over the oblique forest and 0.010 over the tuned tree and logistic regression, all smaller than the fold-to-fold std (≈0.015), and the corrected paired t-test is not significant against either week-5 model. The tuned ensembles are effectively tied, and all five models end near 0.67–0.68, which suggests the limit is the information in these features rather than the model. **Do the hyperparameters fit each model?** Yes: AdaBoost chose shallow trees (depth 4) with a small learning rate and many rounds, i.e. weak learners combined slowly; the random forest kept deep trees (max_depth 22) and let averaging control variance, only regularising the leaves (min 42 samples); the single tree, with nothing to average its errors, needed heavy regularisation (depth 6, leaves of 101). Tuning mattered most where the defaults overfit (forest gaps of +0.08 and +0.11 fell to ≈+0.01). **What would I report?** If an ensemble is chosen, AdaBoost's nested CV accuracy, 0.682 ± 0.017, not the best-trial score (0.684), which is optimistically biased by the selection; given the tie, logistic regression (0.672 ± 0.013) remains a simpler, more interpretable alternative.

## Result Analysis

The full weekly analysis is in [`result_analysis.md`](result_analysis.md).

