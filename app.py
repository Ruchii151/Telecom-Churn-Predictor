# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------
# 1. Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# ------------------------------------------------------------------
# 2. Load & clean data (same as notebooks)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    # same file name and index handling as Model-Building.ipynb
    df = pd.read_csv("D:/Innomatics/Machine Learning/Projects/Telecommunication Churn/churn/Churn Dataset.csv", index_col="Unnamed: 0")  # [file:10]
    # dataset already cleaned in EDA/model notebook (TotalCharges numeric, no nulls) [file:8][file:10]
    return df

df = load_data()

# ------------------------------------------------------------------
# 3. Build full training pipeline (as in notebook)
# ------------------------------------------------------------------
@st.cache_resource
def build_pipeline(df: pd.DataFrame):
    # X, y definition exactly as notebook: all columns except Churn are features [file:10]
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # train-test split train_size=0.7, random_state=42 (same as notebook) [file:10]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    # separate numeric & categorical (tenure, MonthlyCharges, TotalCharges are numeric) [file:10]
    X_train_num = X_train.select_dtypes(include=["int64", "float64"])
    X_test_num = X_test.select_dtypes(include=["int64", "float64"])
    X_train_cat = X_train.select_dtypes(include="object")
    X_test_cat = X_test.select_dtypes(include="object")

    # scale numeric with StandardScaler (fit on train, transform on both) [file:10]
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_num),
        columns=X_train_num.columns,
        index=X_train_num.index,
    )
    X_test_num_scaled = pd.DataFrame(
        scaler.transform(X_test_num),
        columns=X_test_num.columns,
        index=X_test_num.index,
    )

    # encode categoricals using one-hot encoding via pandas.get_dummies (same logic as notebook) [file:10]
    X_train_cat_ohe = pd.get_dummies(X_train_cat, drop_first=True)
    X_test_cat_ohe = pd.get_dummies(X_test_cat, drop_first=True)

    # align train/test encoded columns
    X_train_cat_ohe, X_test_cat_ohe = X_train_cat_ohe.align(
        X_test_cat_ohe, join="left", axis=1, fill_value=0
    )

    # final scaled + encoded matrices
    X_train_final = pd.concat([X_train_num_scaled, X_train_cat_ohe], axis=1)
    X_test_final = pd.concat([X_test_num_scaled, X_test_cat_ohe], axis=1)

    # balance with SMOTE before modeling (as in notebook) [file:10]
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_final, y_train)

    # helper to train all models (same list as notebook) [file:10]
    def train_all_models(X_tr, X_te, y_tr, y_te):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
        }

        results = []
        reports = {}
        cms = {}

        for name, clf in models.items():
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

            results.append(
                {
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1,
                }
            )
            reports[name] = classification_report(
                y_te, y_pred, output_dict=True
            )
            cms[name] = confusion_matrix(y_te, y_pred)

        results_df = pd.DataFrame(results).sort_values(
            "F1-Score", ascending=False
        ).reset_index(drop=True)

        best_model_name = results_df.iloc[0]["Model"]
        best_clf = models[best_model_name]

        return results_df, best_model_name, best_clf, reports, cms

    results_df, best_name, best_clf, reports, cms = train_all_models(
        X_res, X_test_final, y_res, y_test
    )  # same evaluation style as notebook [file:10]

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_final": X_train_final,
        "X_test_final": X_test_final,
        "X_res": X_res,
        "y_res": y_res,
        "scaler": scaler,
        "cat_columns": X_train_cat.columns.tolist(),
        "results_df": results_df,
        "best_name": best_name,
        "best_model": best_clf,
        "reports": reports,
        "cms": cms,
    }


assets = build_pipeline(df)
best_model = assets["best_model"]
scaler = assets["scaler"]
cat_cols = assets["cat_columns"]

# ------------------------------------------------------------------
# 4. Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "EDA",
        "Model Training",
        "Model Comparison",
        "Predict Churn",
    ],
)

# ------------------------------------------------------------------
# 5. Overview
# ------------------------------------------------------------------
if page == "Overview":
    st.title("📡 Telecom Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What it solves")
        st.write(
            "This app analyzes a telecom churn dataset and builds multiple classifiers "
            "to predict whether a customer will churn or not."
        )
        st.write(
            "- Target variable: **Churn** (yes / no)\n"
            "- Numeric features: tenure, MonthlyCharges, TotalCharges\n"
            "- Many categorical service & contract features (internet, contract, payment, etc.)."
        )

    with col2:
        st.markdown("### Dataset snapshot")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

    st.markdown("### Churn distribution")
    st.bar_chart(df["Churn"].value_counts())

# ------------------------------------------------------------------
# 6. EDA
# ------------------------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.markdown("### Summary statistics (numerical)")
    st.dataframe(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())  # [file:8]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Numerical feature distribution")
        num_feature = st.selectbox(
            "Select numerical feature",
            ["tenure", "MonthlyCharges", "TotalCharges"],
        )
        st.bar_chart(df[num_feature].value_counts().sort_index())

    with col2:
        st.markdown("### Categorical feature distribution")
        cat_feature = st.selectbox(
            "Select categorical feature",
            [
                c
                for c in df.columns
                if c
                not in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
            ]
        )
        st.bar_chart(df[cat_feature].value_counts())

    st.markdown("### Churn vs selected feature")
    feat = st.selectbox(
        "Select feature for churn comparison",
        df.columns.drop("Churn"),
    )
    st.bar_chart(df.groupby(feat)["Churn"].value_counts(normalize=True).unstack(fill_value=0))

# ------------------------------------------------------------------
# 7. Model Training details
# ------------------------------------------------------------------
elif page == "Model Training":
    st.title("🤖 Model Training Pipeline")

    st.markdown("### Steps followed")
    st.write(
        "- Split into 70% train and 30% test with random_state=42 (stratified on Churn).\n"
        "- Separate numerical and categorical features.\n"
        "- Scale numerical features with StandardScaler.\n"
        "- One-hot encode categorical features with pandas.get_dummies (drop_first=True).\n"
        "- Apply SMOTE on the transformed training data to balance classes.\n"
        "- Train multiple models and select the best F1-score model."
    )

    st.markdown("### Transformed training features (sample)")
    st.dataframe(assets["X_train_final"].head())

    st.markdown("### SMOTE-balanced training set shape")
    st.write(f"X_res: {assets['X_res'].shape}, y_res: {assets['y_res'].shape}")

# ------------------------------------------------------------------
# 8. Model Comparison
# ------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("📈 Model Comparison")

    st.markdown("### Evaluation metrics on test data")
    st.dataframe(assets["results_df"])

    st.markdown("### Best model")
    st.write(f"Best model based on F1-score: **{assets['best_name']}**")

    st.markdown("### Detailed report for selected model")
    model_name = st.selectbox(
        "Select model for detailed report",
        list(assets["reports"].keys()),
        index=list(assets["reports"].keys()).index(assets["best_name"]),
    )

    st.json(assets["reports"][model_name])
    st.markdown("#### Confusion matrix")
    st.write(assets["cms"][model_name])

# ------------------------------------------------------------------
# 9. Prediction page (single customer)
# ------------------------------------------------------------------
elif page == "Predict Churn":
    st.title("🔍 Predict Customer Churn")

    st.markdown(
        f"Current deployed model: **{assets['best_name']}** (trained with scaling, encoding, SMOTE)."
    )

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        # Numeric inputs
        with c1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)

        # Categorical inputs from dataset unique values (same categories as training) [file:8][file:10]
        with c2:
            gender = st.selectbox("Gender", sorted(df["gender"].unique()))
            senior = st.selectbox("SeniorCitizen", sorted(df["SeniorCitizen"].unique()))
            partner = st.selectbox("Partner", sorted(df["Partner"].unique()))
            dependents = st.selectbox("Dependents", sorted(df["Dependents"].unique()))
            phone = st.selectbox("PhoneService", sorted(df["PhoneService"].unique()))
            multiple = st.selectbox("MultipleLines", sorted(df["MultipleLines"].unique()))
            internet = st.selectbox("InternetService", sorted(df["InternetService"].unique()))
            onsec = st.selectbox("OnlineSecurity", sorted(df["OnlineSecurity"].unique()))
            onbkp = st.selectbox("OnlineBackup", sorted(df["OnlineBackup"].unique()))
            device = st.selectbox("DeviceProtection", sorted(df["DeviceProtection"].unique()))
            tech = st.selectbox("TechSupport", sorted(df["TechSupport"].unique()))
            tv = st.selectbox("StreamingTV", sorted(df["StreamingTV"].unique()))
            movies = st.selectbox("StreamingMovies", sorted(df["StreamingMovies"].unique()))
            contract = st.selectbox("Contract", sorted(df["Contract"].unique()))
            paperless = st.selectbox("PaperlessBilling", sorted(df["PaperlessBilling"].unique()))
            paymethod = st.selectbox("PaymentMethod", sorted(df["PaymentMethod"].unique()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        # create single-row DataFrame in same raw format as training [file:10]
        single = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [senior],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone],
                "MultipleLines": [multiple],
                "InternetService": [internet],
                "OnlineSecurity": [onsec],
                "OnlineBackup": [onbkp],
                "DeviceProtection": [device],
                "TechSupport": [tech],
                "StreamingTV": [tv],
                "StreamingMovies": [movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless],
                "PaymentMethod": [paymethod],
                "MonthlyCharges": [monthly],
                "TotalCharges": [total],
            }
        )

        # same preprocessing: split numeric/categorical, scale, one-hot, align cols [file:10]
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        single_num = single[num_cols]
        single_cat = single.drop(columns=num_cols)

        single_num_scaled = pd.DataFrame(
            scaler.transform(single_num),
            columns=single_num.columns,
            index=single_num.index,
        )

        single_cat_ohe = pd.get_dummies(single_cat, drop_first=True)
        # align with training columns
        X_train_cat_ohe = assets["X_train_final"].drop(columns=num_cols)
        single_cat_ohe = single_cat_ohe.reindex(
            columns=X_train_cat_ohe.columns, fill_value=0
        )

        single_final = pd.concat([single_num_scaled, single_cat_ohe], axis=1)

        pred = best_model.predict(single_final)[0]
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(single_final)[0][
                list(best_model.classes_).index("yes")
            ]
        else:
            proba = None

        st.markdown("---")
        if pred == "yes":
            if proba is not None:
                st.error(f"Prediction: **High churn risk** (probability: {proba:.2f})")
            else:
                st.error("Prediction: **High churn risk**")
        else:
            if proba is not None:
                st.success(f"Prediction: **Low churn risk** (churn probability: {proba:.2f})")
            else:
                st.success("Prediction: **Low churn risk**")

        st.markdown("#### Entered details")
        st.json(single.to_dict(orient="records")[0])