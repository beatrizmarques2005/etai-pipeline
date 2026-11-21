# Result Analysis

`python main.py` scores one split (`random_state=42`). Weeks 2–3 therefore report the **mean ± std over 15 stratified 80/20 splits** (same splits for every configuration). From Week 4 on, scores are **stratified 5-fold CV on the development set** (the locked test set is never used); tuned scores are **nested CV** (30 Optuna trials, 5 inner folds, 5 outer folds). "Gap" = train − validation score.

## Summary

| Week | Main change | Best model | Score |
|---|---|---|---|
| 2 | Baseline pipeline | Logistic Regression | 0.672 ± 0.011 acc. |
| 3 | EDA-driven preprocessing | Decision Tree | 0.675 ± 0.011 acc. |
| 4 | Locked test set + 5-fold CV | Decision Tree | 0.675 ± 0.018 acc. (CV) |
| 5 | Hyperparameter tuning | _to be filled in_ | |
| 6 | Ensembles | AdaBoost | 0.682 ± 0.017 acc. (nested CV) |
| 7 | Boosting + ROC-AUC | HistGradientBoosting | 0.728 ± 0.016 ROC-AUC (nested CV) |
| 8 | XGBoost + feature selection | XGBoost, no selection | 0.728 ± 0.015 ROC-AUC (nested CV) |
| 10 | Imputers | median kept | 0.721 ROC-AUC |

---

## Week 2 — Baseline

Rows with missing values dropped; `pd.get_dummies()` on text columns.

| Model | Accuracy |
|---|---:|
| Logistic Regression | **0.672 ± 0.011** |
| Decision Tree | 0.662 ± 0.014 |

- `dropna()` removed 14% of rows, mostly older defendants.
- Race labels were not cleaned, so one group appeared under several spellings in the fairness check.

**Takeaway:** Logistic Regression is the baseline to beat.

## Week 3 — EDA and preprocessing

Category cleanup, domain-rule checks, duplicate removal, median/mode imputation with MNAR flags, redundant columns dropped, target encoding, standard scaling.

| Model | Accuracy | vs Week 2 |
|---|---:|---:|
| Logistic Regression | 0.670 ± 0.014 | −0.002 |
| Decision Tree | **0.675 ± 0.011** | +0.013 |

- **Other imputers:** see Week 10.
- **New data issue:** 6 rows had an `age_cat` contradicting `age` (now recomputed from `age`); 10 `score_text`/`decile_score` mismatches flagged (neither is a model feature).
- Preprocessing helped the tree and left logistic regression unchanged. False-positive rates fell for both, mainly because more rows were kept and categories cleaned.

**Takeaway:** Decision Tree is best, by a small margin.

## Week 4 — Preprocessing inside the pipeline, cross-validation

- Locked test set: 20%, stratified, seed 42, 1,443 rows. Development set: 5,771 rows.
- Target encoding now cross-fits (`TargetEncoder`); robust scaling replaces standard scaling.
- 229 `c_charge_degree` gaps hidden as the string `"nan"` are now imputed and flagged.

| Model | CV accuracy | Mean gap | F1 (class 1) | vs Week 3 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.672 ± 0.013 | +0.003 | 0.58 | +0.002 |
| Decision Tree | **0.675 ± 0.018** | +0.010 | 0.60 | 0.000 |
| Random Forest (untuned, 300 trees) | 0.650 ± 0.018 | **+0.083** | 0.60 | – |

_"vs Week 3" is indicative only (15 splits of all rows vs 5-fold CV on the development set)._

- Logistic Regression and the tree are tied (0.003 apart, under one std). Logistic Regression is the most stable (smallest gap, std ± 0.013).
- The untuned forest overfits: 0.733 on training folds vs 0.650 on validation (gap 0.083, range 0.068–0.109 across folds). It needs limits such as `max_depth` or `min_samples_leaf`.
- **Fairness (out-of-fold FPR, African-American / Caucasian):** Logistic Regression 0.26 / 0.13 (lowest), Random Forest 0.36 / 0.23 (highest), COMPAS 0.45 / 0.23.

**Takeaway:** Decision Tree, 0.675 ± 0.018, but its lead is smaller than its std.

## Week 5 — Hyperparameter tuning

_To be filled in._ Optuna tuning with nested CV was introduced; results for all models are in the Week 6 and 7 tables.

## Week 6 — Ensembles

Accuracy, 5-fold CV on the development set.

| Model | Default CV (gap) | Tuned nested CV (gap) | Best trial | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | `max_iter=2000` |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.683 | `n_estimators=200`, `max_depth=22`, `min_samples_leaf=42`, `max_features=0.5` |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | `n_estimators=200`, `tree__max_depth=6`, `tree__min_samples_leaf=3`, `project__n_components=5` |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | `estimator__max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

Corrected paired t-test (Nadeau & Bengio, `test_train_ratio=1/4`), tuned AdaBoost vs tuned decision tree: +0.010, 4/5 folds, t = 1.68, p = 0.17; vs logistic regression: +0.010, 3/5 folds, t = 1.01, p = 0.37.

- **Is AdaBoost's lead real?** No. Its nested accuracy (0.682) leads the random forest by about 0.001, the oblique forest by 0.003 and the tuned tree and logistic regression by 0.010, all below the fold std (≈0.015). All five models end near 0.67–0.68: the limit is the information in the features, not the model.
- **Hyperparameters fit each model:** AdaBoost uses shallow trees (depth 4) with a small learning rate and many rounds; the forest keeps deep trees (depth 22) and regularises leaves (min 42); the single tree needs heavy regularisation (depth 6, leaves of 101).

**Takeaway:** the models tie around 0.67–0.68; AdaBoost is nominally best.

## Week 7 — Boosting and metric evaluation

- New models: `gradient_boosting`, `hist_gradient_boosting`, `naive_bayes`.
- Every run reports 8 metrics from out-of-fold **probabilities**; **ROC-AUC** is the selection metric (tuning maximises it, runs are ranked by it).
- Each run adds a row to `results/runs.csv`, next to the no-model floor (dummy, same folds): **0.500**.
- Gradient boosting became the default model. Board saved in [`data/prerun/week7_runs.csv`](data/prerun/week7_runs.csv).

### Scoreboard (ROC-AUC)

| Model | Untuned CV (gap) | Tuned nested CV (gap) | Best trial | Chosen hyperparameters |
|---|---|---|---|---|
| Dummy (floor) | 0.500 ± 0.000 | — | — | — |
| Logistic regression | 0.718 ± 0.014 (+0.002) | — | — | `max_iter=2000` |
| Naive Bayes | 0.685 ± 0.011 (+0.002) | 0.705 ± 0.012 (+0.001) | 0.705 | `var_smoothing=0.091` |
| Decision tree | 0.610 ± 0.020 (+0.107) | 0.717 ± 0.017 (+0.015) | 0.720 | `max_depth=5`, `min_samples_leaf=115`, `criterion=gini` |
| Random forest | 0.684 ± 0.020 (+0.120) | 0.725 ± 0.015 (+0.020) | 0.728 | `n_estimators=200`, `max_depth=24`, `min_samples_leaf=34`, `max_features=0.5` |
| Oblique random forest | 0.685 ± 0.016 (+0.150) | 0.726 ± 0.015 (+0.021) | 0.728 | `n_estimators=200`, `tree__max_depth=12`, `tree__min_samples_leaf=87`, `project__n_components=8` |
| AdaBoost | 0.723 ± 0.016 (+0.005) | 0.727 ± 0.015 (+0.013) | 0.729 | `estimator__max_depth=4`, `n_estimators=55`, `learning_rate=0.167` |
| Gradient boosting | 0.727 ± 0.015 (+0.022) | 0.727 ± 0.014 (+0.012) | 0.729 | `n_estimators=426`, `learning_rate=0.012`, `max_depth=3`, `min_samples_leaf=2`, `subsample=0.77` |
| **HistGradientBoosting** | 0.717 ± 0.020 (+0.058) | **0.728 ± 0.016** (+0.012) | 0.729 | `max_iter=92`, `learning_rate=0.095`, `max_leaf_nodes=6`, `min_samples_leaf=84`, `l2_regularization=0.0008` |

### All 8 metrics

Fold means; nested CV when tuned; threshold 0.5 for threshold-dependent metrics.

| Model | Accuracy | Bal. acc. | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 0.549 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.451 |
| Logistic regression | 0.672 | 0.658 | 0.684 | 0.510 | 0.584 | 0.537 | 0.718 | 0.673 |
| Naive Bayes | 0.621 | 0.587 | 0.743 | 0.245 | 0.368 | 0.282 | 0.705 | 0.656 |
| Decision tree | 0.675 | 0.665 | 0.666 | 0.562 | 0.609 | 0.580 | 0.717 | 0.669 |
| Random forest | 0.679 | 0.668 | 0.676 | 0.555 | 0.609 | 0.575 | 0.725 | 0.686 |
| Oblique random forest | 0.680 | 0.670 | 0.674 | 0.564 | 0.614 | 0.583 | 0.726 | 0.687 |
| AdaBoost | 0.683 | 0.673 | 0.677 | 0.569 | 0.618 | 0.588 | 0.727 | 0.689 |
| Gradient boosting | 0.681 | 0.670 | 0.678 | 0.557 | 0.611 | 0.578 | 0.727 | 0.693 |
| HistGradientBoosting | 0.684 | 0.674 | 0.681 | 0.565 | 0.617 | 0.585 | 0.728 | 0.692 |

### Paired t-tests: HistGradientBoosting vs each model

Corrected paired t-test (Nadeau & Bengio, `test_train_ratio=1/4`), scoreboard folds.

| vs | Mean diff | Folds won | t | p |
|---|---:|---:|---:|---:|
| Gradient boosting | +0.001 | 3/5 | 0.37 | 0.73 |
| AdaBoost | +0.001 | 2/5 | 0.55 | 0.61 |
| Oblique random forest | +0.002 | 4/5 | 1.35 | 0.25 |
| Random forest | +0.003 | 4/5 | 1.95 | 0.12 |
| Logistic regression | +0.010 | 5/5 | 3.18 | 0.034 |
| Decision tree | +0.012 | 5/5 | 3.49 | 0.025 |
| Naive Bayes | +0.023 | 5/5 | 4.81 | 0.009 |
| Dummy | +0.228 | 5/5 | 21.12 | < 0.001 |

### Why ROC-AUC

It scores how well the model *ranks* defendants without fixing a threshold (the 0.5 cut-off is arbitrary; the real threshold is a policy choice between false positives and false negatives). It also compares directly with COMPAS's 1–10 score, and with a base rate of 0.45 the classes are balanced enough for it.

### Findings

- **Boosters and forests tie.** HistGradientBoosting leads gradient boosting, AdaBoost and both forests by 0.001–0.003, far below the std (≈0.015); none of these tests is significant (p = 0.12–0.73). It beats logistic regression, the tuned tree and Naive Bayes on every fold.
- **Tuning helps the overfitters.** The tree gains +0.107 and the forests +0.04 (untuned gaps +0.11 to +0.15); the boosters gain at most +0.011.
- **Metric matters.** Naive Bayes has the best precision (0.743) but recall of 0.245 at 0.5, so its F1 is poor, yet it is only 0.02 behind on ROC-AUC: it ranks fine but is badly calibrated. Its tuned `var_smoothing` (0.091) sits at the edge of the search space (max 0.1).
- **Fairness (threshold 0.5):** HistGradientBoosting FPR is 0.29 (African-American) / 0.15 (Caucasian) vs COMPAS 0.45 / 0.23, still about twice as high for African-American defendants.
- As in Week 6, every reasonable model ends near 0.72–0.73 ROC-AUC (≈0.68 accuracy).

**Takeaway:** HistGradientBoosting, nested ROC-AUC **0.728 ± 0.016**, tied with gradient boosting and AdaBoost. Report the nested number, not the best-trial 0.729.

## Week 8 — XGBoost and feature selection

- New model `xgboost` (now the model `main.py` runs).
- New pipeline step `VoteSelector`: a random forest, gradient boosting and an L1 logistic regression vote; features with at least `min_votes` votes are kept. It sits between preprocessing and the model and is re-fitted in every fold. With tuning on, `select__min_votes` (1–3) is tuned and assessed by nested CV.
- Two tuned runs, `feature_selection.enabled` false and true (ROC-AUC). Board saved in [`data/prerun/week8_runs.csv`](data/prerun/week8_runs.csv). No-model floor: **0.500**.

### Scoreboard (ROC-AUC)

| Model | Untuned CV (gap) | Tuned nested CV (gap) | Best trial | `min_votes` | Features kept |
|---|---|---|---|---|---|
| XGBoost, no selection | 0.724 ± 0.017 (+0.034) | **0.728 ± 0.015** (+0.011) | 0.729 | — | all 10 |
| XGBoost + `VoteSelector` | 0.721 ± 0.013 (+0.027) | 0.725 ± 0.013 (+0.005) | 0.726 | 1 (chosen in all 5 outer folds and on all development rows) | 2 of 10: `age`, `priors_count` |
| *Week 7 best: HistGradientBoosting* | *0.717 ± 0.020* | *0.728 ± 0.016* | *0.729* | — | all 10 |

Chosen hyperparameters:
- No selection: `n_estimators=94`, `learning_rate=0.108`, `max_depth=7`, `min_child_weight=28`, `subsample=0.85`, `colsample_bytree=0.56`, `reg_lambda=0.24`, `reg_alpha=0.002`, `gamma=4.1`.
- With selection: `n_estimators=92`, `learning_rate=0.23`, `max_depth=3`, `min_child_weight=1.8`, `subsample=0.57`, `colsample_bytree=0.63`, `reg_lambda=0.08`, `reg_alpha=0.05`, `gamma=4.0`, `min_votes=1`.

Other tuned metrics (nested CV), accuracy / balanced accuracy / F1 / PR-AUC: without selection 0.680 / 0.670 / 0.614 / 0.692; with selection 0.678 / 0.668 / 0.612 / 0.689.

### Corrected paired t-tests

Nadeau & Bengio, `test_train_ratio=1/4`, same folds.

| Comparison (a vs b) | Mean diff (a − b) | Folds a wins | t | p |
|---|---:|---:|---:|---:|
| Tuned XGBoost **with** vs **without** selection | −0.003 | 2/5 | −0.89 | 0.42 |
| Untuned XGBoost with vs without selection | −0.003 | 2/5 | −0.46 | 0.67 |
| Tuned XGBoost without selection vs Week 7 HistGradientBoosting | −0.0003 | 2/5 | −0.23 | 0.83 |
| Tuned XGBoost without selection vs Week 7 gradient boosting | +0.0004 | 3/5 | 0.28 | 0.79 |
| Tuned XGBoost with selection vs Week 7 HistGradientBoosting | −0.003 | 2/5 | −0.89 | 0.42 |

### What `VoteSelector` keeps

- On all development rows only `age` and `priors_count` get votes (3 of 3 each); every other feature gets 0, except `age_cat` with 1 vote in fold 2.
- So `min_votes` 1, 2 and 3 are almost the same choice. Untuned XGBoost scores 0.7206 / 0.7207 / 0.7208 for `min_votes` 1 / 2 / 3. The nested CV picked 1 in all five folds, which is a tie rather than a real preference.

### Findings

- **XGBoost ties with the other boosters.** Tuned XGBoost (0.728 ± 0.015) vs HistGradientBoosting (0.728), gradient boosting (0.727), AdaBoost (0.727): differences ≤ 0.0004, p = 0.79–0.83.
- **Selection did not improve the score, but cost little.** 0.725 vs 0.728 (p = 0.42), within fold noise (std ≈ 0.015). The test cannot prove equality, but 2 features are clearly *almost as good* as 10.
- **`age` and `priors_count` carry nearly all the usable signal.** A smaller model is easier to explain and cheaper to run.
- **Selection shrinks the gap** (+0.005 vs +0.011 tuned; +0.027 vs +0.034 untuned) and the std a little (0.013 vs 0.015): a small effect.
- **Why it helped less than in the notebook:** there, 20 noise columns were added and selection recovered the score. COMPAS has 10 columns and none is noise, so removing 8 weak-but-real ones costs a little signal.
- **Fairness (threshold 0.5):** FPR 0.30 (African-American) / 0.16 (Caucasian) without selection, 0.30 / 0.17 with it, vs COMPAS 0.45 / 0.23. Selection does not change the picture: the gap comes from how `age` and `priors_count` relate to race in the data.

**Keep selection on?** A judgement call. It gives an equally good model (within noise) with 2 features instead of 10, at about three extra model fits per fold; without it the score is 0.003 higher, not significantly. I keep it on (the config switch turns it off) and do not claim it improves accuracy.

**Takeaway:** tuned XGBoost without selection, nested ROC-AUC **0.728 ± 0.015**, tied with HistGradientBoosting (0.728), gradient boosting and AdaBoost. XGBoost with selection (0.725 ± 0.013, 2 features) is the simplest equally good option.

## Week 10 — Missing values

XGBoost (selection on, tuning off), same folds, four numeric imputers (ROC-AUC):

| Imputer | ROC-AUC | p vs median |
|---|---:|---:|
| median | 0.721 ± 0.013 | — |
| KNN | 0.720 ± 0.012 | 0.86 |
| iterative (MICE) | 0.721 ± 0.014 | 0.95 |
| random forest | 0.721 ± 0.013 | 0.82 |

**Takeaway:** no imputer beats median (differences ≤ 0.0003 AUC, far below fold noise). Median is kept: simplest and fastest.
