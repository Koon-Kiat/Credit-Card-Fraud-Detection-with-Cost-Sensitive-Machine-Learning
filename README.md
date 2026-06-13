# Credit Card Fraud Detection with Cost-Sensitive Machine Learning

An end-to-end machine learning study for detecting fraudulent credit card
transactions while balancing fraud recovery against the operational cost of
false alarms.

## Overview

This project builds and evaluates a leakage-safe fraud detection workflow using
one million simulated credit card transactions. It explores data quality,
imbalanced classification, model selection, controlled ablation experiments,
failure patterns, and business-driven threshold optimization.

The final solution uses a Random Forest classifier with SMOTE and a custom
decision threshold of `0.3`. On the held-out test set, it achieved:

| Metric | Result |
|---|---:|
| Fraud recall | 99.98% |
| Precision | 99.96% |
| F1 score | 99.97% |
| Fraud cases detected | 17,477 / 17,481 |
| Missed fraud cases | 4 |
| False alarms | 7 |

> These results come from a clean, simulated dataset with highly separable
> classes. They should not be treated as expected real-world production
> performance.

## Project Highlights

- Analyzes feature relationships and removes a non-informative feature.
- Uses an 80/20 stratified train-test split.
- Prevents data leakage by keeping imputation, scaling, and resampling inside
  formal Scikit-learn and Imbalanced-learn pipelines.
- Compares SMOTE, partial SMOTE, ADASYN, SMOTE with Tomek Links, and class
  weighting.
- Evaluates Logistic Regression, K-Nearest Neighbors, Gradient Boosting, and
  Random Forest using five-fold stratified cross-validation.
- Performs controlled feature, resampling, and hyperparameter ablations.
- Investigates false negatives and false positives using out-of-fold
  predictions.
- Optimizes the classification threshold using estimated fraud-loss and review
  costs.
- Reserves the hold-out test set for a single final evaluation.

## Dataset

The notebook uses the
[Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
published on Kaggle by Dhanush Narayanan R.

The dataset contains 1,000,000 simulated transactions, with approximately
8.74% labeled as fraud.

| Feature | Description |
|---|---|
| `distance_from_home` | Distance between the cardholder's home and the transaction |
| `distance_from_last_transaction` | Distance from the previous transaction |
| `ratio_to_median_purchase_price` | Purchase amount relative to the cardholder's median purchase |
| `repeat_retailer` | Whether the retailer was previously visited |
| `used_chip` | Whether chip authentication was used |
| `used_pin_number` | Whether a PIN was used |
| `online_order` | Whether the purchase was made online |
| `fraud` | Binary target indicating a fraudulent transaction |

The dataset is not included in this repository. Download
`card_transdata.csv` from Kaggle and place it at:

```text
dataset/card_transdata.csv
```

## Methodology

### 1. Data Preparation

- Removes duplicate rows.
- Drops `repeat_retailer` after exploratory analysis found negligible
  correlation with fraud.
- Imputes continuous values with `KNNImputer`.
- Scales continuous features with `RobustScaler`.
- Imputes binary features with their most frequent value.

### 2. Imbalance Handling

The notebook compares:

- Full SMOTE at a 1:1 class ratio
- Partial SMOTE at approximately a 3.3:1 ratio
- ADASYN
- SMOTE with Tomek Links
- Class-weighted learning

### 3. Model Selection

Four model families are evaluated with five-fold stratified cross-validation:

| Model | Mean fraud recall |
|---|---:|
| Logistic Regression | 94.93% |
| K-Nearest Neighbors | 99.43% |
| Gradient Boosting | 99.89% |
| Random Forest | 99.99% |

Random Forest was selected as the champion model based on recall and
cross-validation stability.

### 4. Threshold Optimization

Because a missed fraudulent transaction is assumed to cost substantially more
than reviewing a false alarm, the operating threshold is reduced from `0.5`
to `0.3`.

Using the notebook's assumptions of `$500` per missed fraud and `$5` per false
alarm review, the tuned threshold reduced estimated hold-out cost from `$3,020`
to `$2,035`.

## Project Structure

```text
Credit-Card-Fraud-ML/
|-- dataset/
|   `-- card_transdata.csv    # Download separately from Kaggle
|-- main.ipynb                # Complete analysis and modeling workflow
`-- README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Credit-Card-Fraud-ML
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

4. Download the Kaggle dataset and place `card_transdata.csv` inside the
   `dataset/` directory.

5. Start Jupyter and run the notebook from top to bottom:

```bash
jupyter notebook main.ipynb
```

## Tools and Libraries

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn

## Limitations

- The dataset is simulated and substantially cleaner than real transaction
  data.
- Transactions are treated as independent observations; temporal and
  cardholder-level behavior is not modeled.
- The notebook demonstrates model development and evaluation, not a deployed
  real-time fraud detection service.
- Cost estimates are illustrative and should be replaced with values specific
  to the organization before operational use.
