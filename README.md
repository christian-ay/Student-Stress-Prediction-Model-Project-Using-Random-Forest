# Student Stress Prediction

**Predicting student mental stress levels (0–5) using behavioral data**

> A compact, reproducible scikit-learn pipeline and notebook that cleans and explores a student behavior dataset, engineers domain features, trains a balanced Random Forest, and exports a single pipeline artifact for inference.

---

## Table of contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Repository Structure](#repository-structure)
* [Getting started](#getting-started)

  * [Requirements](#requirements)
  * [Installation](#installation)
* [Data](#data)
* [What I implemented (step-by-step)](#what-i-implemented-step-by-step)

  * [Cleaning & EDA](#cleaning--eda)
  * [Feature engineering](#feature-engineering)
  * [Preprocessing pipeline](#preprocessing-pipeline)
  * [Modeling & tuning](#modeling--tuning)
  * [Evaluation & persistence](#evaluation--persistence)
* [How to run](#how-to-run)
* [Example: load the saved model and predict](#example-load-the-saved-model-and-predict)
* [Reproducibility notes](#reproducibility-notes)
* [Next steps / Improvements](#next-steps--improvements)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Project Overview

This project demonstrates a practical machine-learning pipeline that predicts students' stress levels (ordinal target values 0–5) from a behavioral dataset. The goal was not only to maximize predictive performance but to produce a clean, reproducible artifact (a single scikit-learn `Pipeline`) that bundles preprocessing and model into one object ready for inference.

You can use this repo as a starting point for behavioral analytics, mental-health risk scoring prototypes, or pedagogy-focused feature engineering examples.

---

## Key Features

* End-to-end Jupyter Notebook showing the entire workflow.
* Domain-driven feature engineering (e.g., **SME** and **sleep\_to\_screen\_ratio**).
* `ColumnTransformer` + `Pipeline` to ensure identical preprocessing at train and inference time.
* Class imbalance handled with `class_weight='balanced'` during training and `StratifiedKFold` CV.
* Hyperparameter search using `RandomizedSearchCV` with `f1_weighted` scoring.
* Model persistence (pickled pipeline) for one-step inference.

---

## Repository Structure

```
student-stress-prediction/
├─ data/
│  └─ student_distress_dataset.csv
├─ notebooks/
│  └─ student_stress_analysis.ipynb
├─ models/
│  └─ mental_stress_pipeline.pkl
├─ requirements.txt
├─ README.md
└─ .gitignore
```

Adjust names to match your repository if you move files around.

---

## Getting started

### Requirements

* Python 3.13
* Recommended: create a virtual environment (conda or venv)

Suggested packages (also included in `requirements.txt`):

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
imbalanced-learn
xgboost  # optional
shap     # optional, for explainability
```

### Installation

```bash
# create and activate virtualenv (example with venv)
python -m venv .venv
source .venv/bin/activate   # mac/linux
.\.venv\Scripts\activate  # windows

# install deps
pip install -r requirements.txt
```

---

## Data

Place your dataset CSV in the `data/` folder (default path: `data/student_distress_dataset.csv`). The notebook expects typical behavioral columns such as:

* `Student ID` (dropped to protect privacy)
* `Gender`, `Age`
* `Social Media Usage` (hours/day)
* `Number of Notifications` (per day)
* `Sleep Duration` (hours/night)
* ...and a target column: `Stress Level` (0–5)

If your columns are named differently, update the notebook or the small wrapper scripts accordingly.

---

## What I implemented (step-by-step)

### Cleaning & EDA

* Removed identifier columns to avoid leakage.
* Normalized or imputed `Prefer not to say` as `pd.NA` then imputed with mode for categorical fields.
* Checked for duplicates and missing values, plotted distributions and correlations.

### Feature engineering

Two domain features were engineered and used in training:

* **SME (Social Media Engagement)**

  * Combines time-on-platform and notification load: `SME = social_media_hours + (notifications / max_notifications)`
  * Purpose: capture both duration and engagement pressure.
* **sleep\_to\_screen\_ratio**

  * `sleep_to_screen_ratio = sleep_hours / (SME + 1)`
  * Purpose: measure restorative sleep relative to screen engagement.

Using these condensed features helped reduce collinearity and boosted signal during training.

### Preprocessing pipeline

* Numeric features: `StandardScaler`.
* Categorical features: `OneHotEncoder(drop='first', handle_unknown='ignore')`.
* Bundled with `ColumnTransformer` so preprocessing is deterministic and reproducible.

### Modeling & tuning

* Base model: `RandomForestClassifier(random_state=42, class_weight='balanced')`.
* Hyperparameter tuning: `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` and `scoring='f1_weighted'`.
* Why these choices?

  * `class_weight='balanced'` addresses class imbalance without synthetic sampling.
  * `StratifiedKFold` preserves class distributions across folds.
  * `f1_weighted` balances precision & recall across all classes.

### Evaluation & persistence

* Evaluation metrics: per-class precision/recall/f1 (via `classification_report`), confusion matrix heatmap.
* Persisted the *entire pipeline* to `models/mental_stress_pipeline.pkl` using `joblib` / `pickle` for easy inference.

---

## How to run

Open the notebook and run cells in order (recommended):

```bash
# start jupyter lab / notebook
jupyter lab    # or jupyter notebook
```

Or run a script (if you add one):

```bash
python src/train.py --data data/student_distress_dataset.csv --out models/mental_stress_pipeline.pkl
```

> The provided notebook contains the exact code used to reproduce preprocessing, CV, and model persistence.

---

## Example: load the saved model and predict

```python
import pickle
import pandas as pd

# load
with open('models/mental_stress_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

# example input -- ensure same engineered fields or raw fields depending on how pipeline is saved
sample = pd.DataFrame([
    {
        'Gender': 'Male',
        'Age': 21,
        'SME': 2.5,
        'sleep_to_screen_ratio': 1.2,
        # ...other fields required by pipeline
    }
])

pred = pipe.predict(sample)
print('Predicted stress level:', pred[0])

# probability (if available)
print('Probabilities:', pipe.predict_proba(sample))
```

---

## Reproducibility notes

* Use the same random seed used inside the notebook (e.g., `random_state=42`).
* Keep the `Pipeline` artifact; it contains preprocessing transforms, so you won’t accidentally preprocess differently at inference time.
* If you change feature names, update the notebook and the saved pipeline accordingly.

---

## Next steps / Improvements

* Add SHAP or permutation importance to explain model outputs.
* Try gradient-boosted models (XGBoost/LightGBM) and calibrated probabilities.
* Add a small REST API (Flask/FastAPI) that loads the pipeline and serves predictions.
* Add monitoring to detect data drift and trigger re-training.

---

## Contributing

Feel free to open issues or PRs. If you find bugs or want a feature (e.g., Dockerfile, REST API, or a CLI wrapper), open an issue and I’ll prioritize.

---

## License

This repo is released under the **MIT License** — see `LICENSE`.

---

## Contact

If you want the full notebook cleaned up into a runnable script, a Dockerfile, or a demo API, ping me here or open an issue.

Happy modeling 🙌 — and remember: data without good features is like coffee without caffeine: wasted potential.


