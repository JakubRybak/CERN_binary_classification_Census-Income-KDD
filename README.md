# Census Income Prediction Project

## 1. Project Overview
This project focuses on a binary classification problem to predict whether an individual earns more than $50,000 annually. The goal is to analyze demographic and employment data, perform feature engineering, and optimize a Support Vector Machine (SVM) model to maximize predictive performance while maintaining model fairness and interpretability.

## 2. Dataset and Assigned Model
*   **Dataset:** Census-Income-KDD (approx. 200,000 records).
*   **Target Variable:** `income_50k` (Binary: 0 or 1).
*   **Assigned Model:** Support Vector Machine (SVM).

## 3. Methodology
The project followed a rigorous pipeline focusing on data quality and ethical feature selection:

### A. Feature Engineering & Selection
*   **Engineered Features:** Created custom features such as `net_capital` and `is_investor` to better capture financial status. These proved to be among the most important predictors.
*   **No PCA:** Principal Component Analysis (PCA) was consciously **avoided** to preserve feature interpretability. Using raw (but scaled) features allowed for precise impact analysis using SHAP values.
*   **Ethical Constraints (Fairness):** Sensitive attributes—specifically **`sex`**, **`race`**, and **`hisp_origin`**—were excluded from the final model. Preliminary experiments showed that including these features provided negligible improvement in metrics, so they were removed to avoid algorithmic bias.

### B. Model Optimization
*   **Process:** Initial training was performed on a 5k subset for hyperparameter tuning, followed by validation on larger sets.
*   **Experiments:** Tested various SVM kernels (Linear, RBF), class weighting strategies, and undersampling.
*   **Benchmark:** An **XGBoost** model was trained to benchmark the SVM. The tree-based model yielded similarly moderate results, suggesting that the performance ceiling is due to intrinsic data difficulty rather than the choice of the SVM algorithm.

## 4. Key Findings
*   **Data Complexity:** The classification task is inherently difficult due to weak direct predictors for high income in the dataset. Both SVM and XGBoost struggled to surpass an F1-score of 0.60.
*   **Feature Importance (SHAP):**
    *   As shown in the SHAP summary plot, **`net_capital`** and **`weeks_worked`** are the most influential features.
    *   The high impact of custom features (`net_capital`, `is_investor`) validates the feature engineering strategy.
*   **Metric Analysis:**
    *   **High ROC AUC (0.93):** The model is very good at ranking individuals (separating classes in probability space).
    *   **Moderate F1 (0.58):** Despite good ranking, finding a precise decision boundary is challenging due to the imbalance and noise, resulting in lower precision/recall trade-offs.

## 5. Final Model & Metrics
The final model uses an RBF kernel, optimized for the best balance between detecting high earners and minimizing false positives.

**Model Configuration:**
*   **Kernel:** RBF
*   **C:** 800
*   **Gamma:** 0.0009
*   **Support Vector Ratio:** ~8%

**Test Set Metrics:**

| Metric | Score |
| :--- | :--- |
| **ROC AUC** | **0.93** |
| **Recall** | 0.61 |
| **F1-Score** | 0.58 |
| **Precision** | 0.55 |
