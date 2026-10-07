# Result Analysis

## How results are evaluated

`python main.py` evaluates one train/test split (`random_state=42`). A single split can favour
one model by chance, so each comparison below also reports the **mean ± standard deviation over
15 stratified 80/20 splits**, with each split identical across all configurations.

---

## Weekly Experiments
Accuracy is the mean ± std over 15 stratified 80/20 splits, unless stated otherwise.

### Summary
---

| Week | Main change | Best model | Mean test accuracy |
|---|---|---|---:|
| 2 | Baseline pipeline | Logistic Regression | 0.672 ± 0.011 |
| 3 | EDA-driven preprocessing | Decision Tree | 0.675 ± 0.011 |
| 4 | Locked test set + 5-fold CV | Decision Tree | 0.675 ± 0.018 (5-fold CV) |

### Week 2 — Baseline
---

**What changed:** rows with missing values dropped, `pd.get_dummies()` on text columns.

| Model | Mean test accuracy |
|---|---:|
| Logistic Regression | **0.672 ± 0.011** |
| Decision Tree | 0.662 ± 0.014 |

**Findings:** `dropna()` removed 14% of the rows, mostly older defendants. Race labels were not cleaned, so the fairness check split the same group into several spellings.

**Best model:** Logistic Regression.

### Week 3 — EDA and preprocessing
---

**What changed:** category cleanup, domain-rule checks, duplicate removal, median/mode imputation with MNAR flags, redundant columns dropped, target encoding and standard scaling.

| Model | Mean test accuracy | vs Week 2 |
|---|---:|---:|
| Logistic Regression | 0.670 ± 0.014 | −0.002 |
| Decision Tree | **0.675 ± 0.011** | +0.013 |

**Challenge answers:**

1. **Other imputers:** ...
2. **New data issue:** 6 rows had an `age_cat` that contradicted `age`. These are now recomputed from `age`. 10 `score_text`/`decile_score` mismatches were flagged; neither column is a model feature.
3. **Week 2 vs Week 3:** preprocessing improved the decision tree and left logistic regression about the same. False-positive rates fell for both models, mainly because more rows were kept and categories were cleaned. 

**Best model:** Decision Tree.

### Week 4 — Preprocessing inside the pipeline and cross-validation
---

**What changed:** a locked final test set (20%, stratified, seed 42, 1,443 rows) is set aside and never scored.
Models are now judged by stratified 5-fold cross-validation of the whole pipeline (preprocessing + model) on the
5,771 development rows. 

**Other changes**: target encoding switched to scikit-learn's cross-fitting `TargetEncoder`,
standard scaling replaced by robust scaling, 229 `c_charge_degree` gaps that had been hidden as the string `"nan"`
are now imputed and flagged.

| Model | CV val. accuracy (mean ± std) | Mean train–val gap | F1 (class 1) | vs Week 3 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.672 ± 0.013 | +0.003 | 0.58 | +0.002 |
| Decision Tree | **0.675 ± 0.018** | +0.010 | 0.60 | 0.000 |
| Random Forest (untuned, 300 trees) | 0.650 ± 0.018 | **+0.083** | 0.60 | – |

*"vs Week 3" is only indicative: Week 3 used 15 repeated 80/20 splits of all rows, Week 4 uses 5-fold CV on the
development set only.*

**Findings:**

- **Logistic regression and the decision tree are tied.** They differ by 0.003, well under one standard deviation, and neither changed meaningfully from Week 3. 
- **The untuned random forest overfits.** It scores 0.733 on the training folds but only 0.650 on validation, a gap of 0.083 in every fold (0.068–0.109). With default settings each tree grows until it memorises its training rows, so it needs limits such as `max_depth` or `min_samples_leaf` before it can compete.
- **Logistic regression is the most stable model.** It has the smallest gap (+0.003) and the lowest spread across folds (± 0.013).
- **Fairness:** on out-of-fold predictions, logistic regression has the lowest false-positive rates (African-American 0.26, Caucasian 0.13), well below COMPAS's own score (0.45 / 0.23). The random forest has the highest (0.36 / 0.23).

**Best model:** Decision Tree, 0.675 ± 0.018, by a margin smaller than its standard deviation.

### Week 5 — Hyperparameter tunning
---

**What changed:** ...

| Model | Mean test accuracy | vs Week 4 |
|---|---:|---:|
| ... | ... | ... |


| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | ... | ... | ... | ... | ... |
| Logistic regression | ... | ... | ... | ... | ... |
| Random forest | ... | ... | ... | ... | ... |

**Challenge answers:**
- Which models gained from tuning, and is the gain larger than the fold-to-fold std?
- Did tuning change which model is best? Which number would you report for your best model, and why?

**Findings:** ...

**Best model:** ...

<!--

## Challenges

Start with 1 and 2 (they fill the README table); the rest are for going further. For every change, keep the rules: development set only, whole pipeline, same folds.

1. **Tune logistic regression.** Add a `logistic_regression` search space to `config.yaml`. Its main hyperparameter is the regularisation strength `C` (smaller = stronger regularisation): search it on a **log scale**, e.g. 0.0001 to 100. Does tuning help LR as much as it helped the tree? Why might a linear model have less to gain?
2. **Tune the random forest.** Add a `random_forest` search space: `n_estimators`, `max_depth` (include `null` - YAML for `None` - as an option: use `categorical`), `min_samples_leaf`, `max_features` (e.g. `["sqrt", "log2", 0.5, 1.0]`). A forest makes every trial ~100× slower: start with `n_trials: 15`, set `cv.n_jobs: -1`, and keep `n_estimators` modest while tuning. Compare the depth the forest chooses with the depth the tree chose - does it match your answer to Section 2.4?

**Going further**:

3. **Tune the preprocessing too.** Add `prep__numeric__impute__strategy: {type: categorical, choices: ["median", "mean"]}` to the tree's search space. Why is this still leak-free? Would it still be leak-free if the preprocessing were fitted once, outside the trials (the caching note in Section 5)?
4. **Change what "best" means.** Set `cv.scoring` to `"balanced_accuracy"` or `"f1"` and tune again. Do the chosen hyperparameters change? What happens to recall of the positive class, and to the false positive rates in the fairness audit?

-->

<!-- TEMPLATE
### Week N — <topic>
---

**What changed:** ...

| Model | Mean test accuracy | vs Week N-1 |
|---|---:|---:|
| ... | ... | ... |

**Findings:** ...

**Best model:** ...
-->
