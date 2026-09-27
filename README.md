# German Credit Risk — Classifying Loan Default Risk

A classification project predicting whether a loan applicant is a good or bad credit risk, using the [German Credit Risk dataset](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) (1,000 applicants, Kaggle). Built end-to-end: data cleaning → EDA → feature engineering → imbalance handling → modeling → SHAP explainability.

**Headline finding:** the strongest predictor of credit risk isn't a financial figure — it's a *missing* one. Applicants with no recorded checking account status are the model's single biggest signal, and they skew toward **lower** risk, not higher.

## Results at a Glance

| | |
|---|---|
| **Final model** | Random Forest (300 trees, `class_weight='balanced'`), threshold = 0.35 |
| **Recall (bad)** | 62.2% |
| **Precision (bad)** | 54.9% |
| **ROC-AUC** | 0.758 |
| **Top predictor** | Missing checking account status → *lower* risk (counterintuitive) |

---

## 1. Data Cleaning

- 1,000 rows, 10 columns (`Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`, `Risk`).
- `Checking account` was missing for **39.4%** of applicants, `Saving accounts` for **18.3%**. Rather than drop nearly 40% of the data, missing values were filled with an explicit `'none'` category — treating "no account on record" as a meaningful state rather than noise.
- 0 duplicate rows.
- Target is imbalanced: **70% good / 30% bad.**

## 2. Exploratory Data Analysis

Three findings drove the modeling decisions below:

| Finding | Detail |
|---|---|
| Checking account status | `little` = 49.3% bad (riskiest); `none` (missing) = **11.7% bad — the lowest of any group**, even lower than `rich` (22.2%) |
| Loan purpose | `vacation/others` (41.7% bad) and `education` (39.0% bad) are riskiest; `radio/TV` is safest (22.1% bad) |
| Age | Bad-risk rate drops steadily from 42.1% (18–25) to 22.2% (60+) — a clear, near-monotonic trend |

The checking-account finding directly shaped feature engineering: since `none` breaks the natural low→high ordering of account balances, it ruled out ordinal encoding in favor of one-hot.

![Checking account status vs. credit risk](images/checking_account_vs_risk.png)

## 3. Feature Engineering

- One-hot encoded `Sex`, `Housing`, `Purpose`, `Saving accounts`, `Checking account` (nominal, not ordinal — see above).
- Engineered `monthly_burden` = `Credit amount / Duration`, a rough proxy for monthly repayment load.
- Binned `Age` into 5 groups (18–25 … 60+) to capture the non-linear pattern found in EDA.

## 4. Handling Class Imbalance & Modeling

70/30 train/test split, stratified. Three models compared:

| Model | Recall (bad) | Precision (bad) | F1 (bad) | ROC-AUC |
|---|---|---|---|---|
| Baseline Logistic Regression | 0.389 | 0.574 | 0.464 | 0.738 |
| Logistic Regression (`class_weight='balanced'`) | 0.689 | 0.484 | 0.569 | 0.741 |
| Random Forest (`class_weight='balanced'`) | 0.389 | 0.729 | 0.507 | **0.758** |

Random Forest had the best ranking power (highest ROC-AUC) but its default 0.5 decision threshold was too conservative to translate that into recall. Sweeping the threshold on its predicted probabilities found the optimal cutoff:

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.50 | 0.400 | 0.706 | 0.511 |
| 0.40 | 0.533 | 0.565 | 0.549 |
| **0.35** | **0.622** | **0.549** | **0.583** ← peak |
| 0.30 | 0.678 | 0.459 | 0.547 |
| 0.25 | 0.767 | 0.429 | 0.550 |

**Final model: Random Forest (300 trees, `class_weight='balanced'`), decision threshold = 0.35** — recall 62.2%, precision 54.9%, F1 0.583, ROC-AUC 0.758.

![Model comparison across baseline, balanced logistic regression, and random forest](images/model_comparison.png)

## 5. Explainability (SHAP)

Top features by mean absolute SHAP value:

| Rank | Feature | Mean \|SHAP value\| |
|---|---|---|
| 1 | `Checking account_none` | **0.109** |
| 2 | `Credit amount` | 0.049 |
| 3 | `Duration` | 0.049 |
| 4 | `monthly_burden` | 0.048 |
| 5 | `Saving accounts_none` | 0.031 |
| 6 | `Age` | 0.027 |
| 7 | `Housing_own` | 0.026 |

`Checking account_none` outweighs every other feature by more than 2x — confirming, with model-level evidence, the counterintuitive pattern first spotted in EDA: a missing checking account is a strong signal of *lower* risk, not higher. Had that missing data simply been dropped in step 1, the model would have lost its single most informative feature.

![Top 10 features by SHAP importance](images/shap_feature_importance.png)

## Limitations & Future Work

- **Small dataset (1,000 rows)** from a single country/time period — the patterns found (e.g. the checking-account effect) may not generalize to other lending markets without re-validation.
- **No hyperparameter tuning** was done on the Random Forest (default depth, fixed `n_estimators=300`) — a grid or random search could likely improve recall further.
- **Threshold was tuned on the test set**, which risks slight overfitting to that specific split; a more rigorous approach would tune it via cross-validation.
- **SMOTE was not used in the final model** — only `class_weight='balanced'` was applied. Comparing against a SMOTE-resampled version would be a natural next experiment.
- Next step: wrap the final model in a simple Flask/FastAPI endpoint or Streamlit demo so it's interactive, not just notebooks.

## Tools

Python, pandas, scikit-learn, SHAP, seaborn/matplotlib.

## How to Run

Notebooks are numbered and meant to be run in order; each one picks up where the last left off:

1. `01_explore_clean_german_credit.ipynb`
2. `02_eda_relationships_german_credit.ipynb`
3. `03_feature_engineering_german_credit.ipynb` — saves `german_credit_model_ready.csv`, used by notebooks 4–5
4. `04_modeling_imbalance_german_credit.ipynb`
5. `05_shap_explainability_german_credit.ipynb`

```bash
pip install -r requirements.txt
```
