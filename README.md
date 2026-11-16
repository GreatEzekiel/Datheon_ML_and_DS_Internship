# Datheon Machine Learning and Data Science Internship

This repository contains four end-to-end machine-learning assignments. Each notebook covers data inspection, exploratory analysis, model training, evaluation, and task-specific conclusions using the supplied CSV files.

## Project files

| Task | Notebook | Dataset | Main methods |
|---|---|---|---|
| 1. Iris classification | [Task_1_Iris.ipynb](./Task_1_Iris.ipynb) | [Task 1-Iris.csv](./Task%201-Iris.csv) | Tuned Logistic Regression and Decision Tree |
| 2. Housing price prediction | [Task_2_Housing.ipynb](./Task_2_Housing.ipynb) | [Task 2-Housing.csv](./Task%202-Housing.csv) | OLS, Ridge, Lasso, and tuned XGBoost regression |
| 3. Customer churn prediction | [Task_3_Customer_Churn_Prediction.ipynb](./Task_3_Customer_Churn_Prediction.ipynb) | [Task 3-Telco-Customer-Churn.csv](./Task%203-Telco-Customer-Churn.csv) | Tuned Random Forest and XGBoost classification |
| 4. Review sentiment classification | [Task_4_NLP_Sentiment.ipynb](./Task_4_NLP_Sentiment.ipynb) | [Task 4-Sentiment_reviews.csv](./Task%204-Sentiment_reviews.csv) | TF-IDF, Multinomial Naive Bayes, and Logistic Regression |

`housing_features.py` contains the shared housing feature-engineering function used by the notebook and its serialized model.

## Environment setup

Use Python 3 and run the notebooks from the repository root so the relative dataset paths resolve correctly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels plotly ipywidgets jupyter ipykernel joblib nltk
jupyter notebook
```

In VS Code, open the repository folder, select the `.venv` Python environment as the notebook kernel, and run each notebook from top to bottom. Task 4 may need an internet connection on its first run to download NLTK stopword and WordNet data; the notebook checks for these resources and requests them if missing.

## Notebook results

The following values are measured on the notebooks' fixed train/test splits. They are reproducible for the current data and settings, but they are not guarantees of future performance.

### Task 1: Iris classification

| Model | 5-fold CV accuracy | Test accuracy |
|---|---:|---:|
| Logistic Regression | 0.9667 | 1.0000 |
| Decision Tree | 0.9667 | 0.9667 |

The Logistic Regression one-vs-rest macro ROC-AUC was 1.0000. The test set contains only 30 samples, so one additional error would change accuracy by about 3.33 percentage points.

### Task 2: Housing price prediction

| Model | Test RMSE | Test MAE | Test R-squared |
|---|---:|---:|---:|
| XGBoost | $42,827.42 | $27,715.81 | 0.8600 |
| OLS | $68,598.40 | $49,442.69 | 0.6409 |
| Ridge | $68,598.41 | $49,442.70 | 0.6409 |
| Lasso | $68,736.68 | $49,546.47 | 0.6394 |

XGBoost was selected by training-only cross-validation RMSE. The target is capped at $500,001 for about 4.68% of rows. The random district split may also overstate performance when predicting new geographic areas; use a spatial holdout for that use case.

### Task 3: Customer churn

The tuned XGBoost model was selected by training cross-validation ROC-AUC (0.8504). On the held-out test set it achieved:

- ROC-AUC: **0.8490**
- Accuracy at threshold 0.50: **0.8048**
- Precision: **0.6689**
- Recall: **0.5241**
- F1-score: **0.5877**

The Random Forest test ROC-AUC was 0.8430. At threshold 0.50 it had higher churn recall (0.7834) but lower precision (0.5298), illustrating the trade-off between missed churners and unnecessary retention outreach.

### Task 4: Sentiment classification

The notebook reports 1.0000 for accuracy, precision, recall, F1-score, and ROC-AUC for both classifiers on its held-out split. **Treat these scores as a demonstration only:** the source file has only 10 reviews, one neutral review is excluded from binary training, and the test set contains just three reviews. A larger labeled dataset and independent validation set are needed for a meaningful performance estimate.

## Generated model and evaluation files

Running the notebooks creates or updates the following artifacts:

- **Task 1:** `iris_best_model.joblib`, `iris_test_metrics.csv`
- **Task 2:** `outputs/housing_best_model.joblib`, and CSV reports under `outputs/model_results/`
- **Task 3:** `customer_churn_best_model.joblib`, `customer_churn_test_metrics.csv`, and `customer_churn_feature_importance.csv`
- **Task 4:** the TF-IDF vectorizer, both classifier files, and `Task 4 - Sentiment Analysis - Model Metrics.csv`

The housing model also requires `housing_features.py` to be available when loading it with `joblib`. For the sentiment artifacts, apply the same text-cleaning and feature-construction steps from the notebook before transforming reviews.
