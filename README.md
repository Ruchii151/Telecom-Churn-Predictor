# Telecom-Churn-Predictor
Telecom Churn Prediction is a supervised classification project that predicts whether a telecom customer will churn based on their demographic, contract, and service usage information.

## Project Overview

This project uses a publicly available telecom churn dataset with 7,000+ customers and 20 cleaned features (3 numeric: tenure, MonthlyCharges, TotalCharges; and multiple categorical service/contract attributes) plus the target Churn (yes/no). The workflow covers EDA, data cleaning, preprocessing, multiple model training, evaluation, and a Streamlit web app for interactive churn prediction.

## Repository Structure

- `Churn-Dataset.csv`: Final cleaned dataset used for EDA and model training.
- `Churn-EDA.ipynb`: Exploratory data analysis, data cleaning, feature understanding, and visual insights on churn drivers.
- `Model-Building.ipynb`: End‑to‑end modeling notebook (encoding, scaling, train/test split, SMOTE, multiple models, evaluation).
- `app.py`: Streamlit application that reproduces the same preprocessing pipeline and serves trained models for live predictions.

## Data and Preprocessing

- Drop `customerID` as an identifier with no predictive value.
- Convert `TotalCharges` from object to numeric and remove duplicate records; confirm no missing values remain.
- Standardize categorical values (lowercasing, consistent labels) and map `SeniorCitizen` from 0/1 to no/yes for readability.
- Separate features:
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`.
  - Categorical: all remaining service, contract, billing, payment and demographic columns.

## Modeling Pipeline

Implemented identically in `Model-Building.ipynb` and `app.py`:

1. **Train–test split**  
   - 70% train, 30% test, `random_state=42`, stratified by `Churn`.

2. **Feature engineering**  
   - Numeric: scale with `StandardScaler` (fit on train, transform train/test).  
   - Categorical: one‑hot encode with `pandas.get_dummies(drop_first=True)`, then align train/test columns.

3. **Class imbalance handling**  
   - Apply `SMOTE(random_state=42)` on the transformed training set only.

4. **Models trained**  
   - Logistic Regression (`max_iter=1000`)  
   - Decision Tree  
   - Random Forest  
   - SVM (`SVC(probability=True)`)  
   - KNN  
   - Gaussian Naive Bayes

5. **Evaluation**  
   - Metrics on test set: Accuracy, weighted Precision, Recall, F1‑Score.
   - Store full `classification_report` and `confusion_matrix` for each model; select the best model by highest F1‑Score.

## Streamlit App Features

The Streamlit app wraps the trained pipeline and best model for interactive use:

- **Overview**  
  - Project description, dataset shape, head of data, churn distribution bar chart.

- **EDA**  
  - Summary statistics for numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).  
  - Selectable distributions for numeric and categorical features.  
  - Churn rate vs selected feature (normalized bar chart).

- **Model Training**  
  - Clear step‑by‑step description of the preprocessing and training pipeline.  
  - Preview of transformed training features.  
  - SMOTE‑balanced train set shape (`X_res`, `y_res`).

- **Model Comparison**  
  - Table of metrics for all models.  
  - Best model name based on F1‑Score.  
  - JSON view of detailed classification report and printed confusion matrix for the selected model.

- **Predict Churn**  
  - Form with:
    - Numeric inputs: `tenure`, `MonthlyCharges`, `TotalCharges`.  
    - Categorical inputs: taken from the dataset’s unique values for all service, contract, and demographic columns.
  - The same preprocessing steps are applied to the single input row:
    - Split numeric/categorical, scale numerics with the saved scaler, `get_dummies` for categoricals, reindex to training columns, then concatenate.
  - Uses the best model to predict `Churn` and, if available, outputs churn probability from `predict_proba` for the `yes` class.
  - Displays “High churn risk” or “Low churn risk” along with the entered input summary.

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

Create `requirements.txt` (example based on used libraries):

```text
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Ensure file paths

Update any hard‑coded paths in `app.py` so the dataset is loaded from the project root (e.g. `Churn-Dataset.csv` in the same folder):

```python
df = pd.read_csv("Churn-Dataset.csv")
```

### 5. Run notebooks (optional but recommended)

Open `Churn-EDA.ipynb` and `Model-Building.ipynb` in Jupyter / VS Code to:

- Reproduce EDA.  
- Re‑run modeling steps and confirm metrics.  
- Optionally export and save a fitted model if you decide to persist instead of retraining inside `app.py`.

### 6. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default: `http://localhost:8501`) and navigate through:

- Overview → EDA → Model Training → Model Comparison → Predict Churn.


# Author
Ruchita Patil Email: pruchita565@gmail.com

LinkedIn Profile: www.linkedin.com/in/patil-ruchita
