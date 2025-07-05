# 🧠 Introvert vs Extrovert Classification with XGBoost

This Jupyter Notebook implements a complete machine learning pipeline to classify individuals as **Introverts** or **Extroverts** using personality and behavioral data. The model is trained with XGBoost and evaluated using cross-validation.

---

## 📦 1. Imports and Setup

Essential libraries are imported, including:

- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualizations
- `sklearn` for preprocessing, evaluation, and model utilities
- `xgboost` for model training
- Warnings are suppressed and random seed is set for reproducibility

---

## 📂 2. Load and Combine Data

- Training and test data are loaded from CSV files.
- An external dataset is also loaded and repeated to balance the training data.
- The data is stored in `train`, `test`, and `external` DataFrames.

---

## 🧹 3. Preprocessing

A custom `preprocess()` function is defined to:

- Drop unnecessary columns (like `id`)
- Encode categorical features using `LabelEncoder`
- Separate features from the target (`Personality`)

It’s applied to:
- Training data (`X_train`, `y_train`)
- Test data (`X_test`)
- External dataset (`X_external`, `y_external`)

The target labels are also encoded to binary format.

---

## ⚙️ 4. XGBoost Parameters

XGBoost hyperparameters are defined, including:

- `learning_rate`, `max_leaves`, `subsample`, `colsample_bytree`, etc.
- GPU support via `tree_method='hist'` and `device='cuda'`
- Set to train up to `10000` estimators with early stopping

---

## 🔁 5. Cross-Validation

A custom `train_with_cv()` function:

- Uses 5-fold stratified cross-validation
- Trains XGBoost models on combined training + external data
- Records accuracy scores and best iteration per fold

Average CV accuracy is printed after training.

---

## 🧠 6. Final Training

- The average best iteration from CV is used to set the final number of estimators.
- The model is retrained on the full training set.
- Final model is stored in `final_model`.

---

## 📊 7. Feature Importance

- Feature importances are extracted from the trained model.
- A barplot shows the top 20 most important features using `seaborn`.

---

## 📤 8. Prediction and Submission

- Predictions are made on the test set.
- The predicted labels are inverse-transformed to their original string labels.
- Results are saved into a CSV file named `submission.csv`.

---

## ✅ Output

- Trained XGBoost model with ~97% cross-validation accuracy
- `submission.csv` file containing the final predictions
- Feature importance visualization for model explainability
