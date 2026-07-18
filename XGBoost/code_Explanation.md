# ============================================================
# Credit Scoring using XGBoost Classifier
# Data Loading & Exploratory Data Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("credit_score.csv")

print("\n")

print("="*70)
print("Dataset Loaded Successfully")
print("="*70)

# ============================================================
# Display First Five Records
# ============================================================

print("\n")

print("="*70)
print("First Five Records")
print("="*70)

display(df.head())

# ============================================================
# Display Last Five Records
# ============================================================

print("\n")

print("="*70)
print("Last Five Records")
print("="*70)

display(df.tail())

# ============================================================
# Dataset Shape
# ============================================================

print("\n")

print("="*70)
print("Dataset Shape")
print("="*70)

print("Rows    :", df.shape[0])

print("Columns :", df.shape[1])

# ============================================================
# Dataset Information
# ============================================================

print("\n")

print("="*70)
print("Dataset Information")
print("="*70)

df.info()

# ============================================================
# Column Names
# ============================================================

print("\n")

print("="*70)
print("Column Names")
print("="*70)

print(df.columns.tolist())

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*70)
print("Statistical Summary")
print("="*70)

display(df.describe())

# ============================================================
# Complete Statistical Summary
# ============================================================

print("\n")

print("="*70)
print("Complete Statistical Summary")
print("="*70)

display(df.describe(include="all"))

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("="*70)
print("Missing Values")
print("="*70)

missing_values = df.isnull().sum()

display(missing_values)

# ============================================================
# Duplicate Records
# ============================================================

print("\n")

print("="*70)
print("Duplicate Records")
print("="*70)

duplicates = df.duplicated().sum()

print("Duplicate Records :", duplicates)

# ============================================================
# Remove Duplicate Records
# ============================================================

df.drop_duplicates(inplace=True)

print("\n")

print("="*70)
print("Duplicate Records Removed Successfully")
print("="*70)

print("Current Shape :", df.shape)

# ============================================================
# Data Types
# ============================================================

print("\n")

print("="*70)
print("Data Types")
print("="*70)

display(df.dtypes)

# ============================================================
# Numerical Columns
# ============================================================

numerical_columns = df.select_dtypes(include=["int64","float64"]).columns

print("\n")

print("="*70)
print("Numerical Columns")
print("="*70)

print(numerical_columns.tolist())

# ============================================================
# Categorical Columns
# ============================================================

categorical_columns = df.select_dtypes(include=["object"]).columns

print("\n")

print("="*70)
print("Categorical Columns")
print("="*70)

print(categorical_columns.tolist())

# ============================================================
# Unique Values
# ============================================================

print("\n")

print("="*70)
print("Unique Values")
print("="*70)

display(df.nunique())

# ============================================================
# Value Counts of Target Variable
# ============================================================

print("\n")

print("="*70)
print("Target Variable Distribution")
print("="*70)

print(df["Credit_Score"].value_counts())

# ============================================================
# Target Variable Percentage
# ============================================================

print("\n")

print("="*70)
print("Target Percentage")
print("="*70)

print(round(df["Credit_Score"].value_counts(normalize=True)*100,2))

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Histogram")
print("="*70)

df.hist(

    figsize=(18,15),

    bins=20

)

plt.tight_layout()

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

print("\n")

print("="*70)
print("Correlation Matrix")
print("="*70)

numeric_df = df.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(14,10))

sns.heatmap(

    numeric_df.corr(),

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# Boxplots
# ============================================================

print("\n")

print("="*70)
print("Boxplots")
print("="*70)

for column in numerical_columns:

    plt.figure(figsize=(8,4))

    sns.boxplot(

        x=df[column]

    )

    plt.title(column)

    plt.show()

# ============================================================
# Pairplot (First Five Numerical Features)
# ============================================================

print("\n")

print("="*70)
print("Pairplot")
print("="*70)

sns.pairplot(

    df[numerical_columns[:5]]

)

plt.show()
# ============================================================
#  Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.preprocessing import LabelEncoder

# ============================================================
# Remove Unnecessary Columns
# ============================================================

columns_to_drop = [

    "ID",

    "Customer_ID",

    "Name",

    "SSN"

]

existing_columns = [column for column in columns_to_drop if column in df.columns]

df.drop(

    columns=existing_columns,

    inplace=True

)

print("\n")

print("="*70)
print("Unnecessary Columns Removed")
print("="*70)

print(df.columns)

# ============================================================
# Convert Numeric Columns Stored as Object
# ============================================================

numeric_columns = [

    "Age",

    "Annual_Income",

    "Monthly_Inhand_Salary",

    "Num_of_Loan",

    "Num_of_Delayed_Payment",

    "Changed_Credit_Limit",

    "Outstanding_Debt",

    "Amount_invested_monthly",

    "Monthly_Balance"

]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"

        )

print("\n")

print("="*70)
print("Numeric Conversion Completed")
print("="*70)

# ============================================================
# Fill Missing Values
# ============================================================

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        # Fill numeric columns with median
        df[column].fillna(df[column].median(), inplace=True)
    else:
        # Fill non-numeric (object, string, category) with mode
        df[column].fillna(df[column].mode()[0], inplace=True)

print("\n")

print("="*70)
print("Missing Values Filled")
print("="*70)

print(df.isnull().sum())

# ============================================================
# Encode Target Variable
# ============================================================

target_encoder = LabelEncoder()

df["Credit_Score"] = target_encoder.fit_transform(

    df["Credit_Score"]

)

print("\n")

print("="*70)
print("Target Encoding Completed")
print("="*70)

print(target_encoder.classes_)

# ============================================================
# Encode Remaining Categorical Columns
# ============================================================

categorical_columns = df.select_dtypes(

    include=["object"]

).columns

label_encoder = LabelEncoder()

for column in categorical_columns:

    df[column] = label_encoder.fit_transform(

        df[column]

    )

print("\n")

print("="*70)
print("Categorical Encoding Completed")
print("="*70)

# ============================================================
# Feature Matrix and Target Variable
# ============================================================

X = df.drop(

    "Credit_Score",

    axis=1

)

y = df["Credit_Score"]

print("\n")

print("="*70)
print("Feature Matrix and Target Variable")
print("="*70)

print("Features Shape :", X.shape)

print("Target Shape   :", y.shape)

# ============================================================
# Display Features
# ============================================================

print("\n")

print("="*70)
print("First Five Rows of Features")
print("="*70)

display(X.head())

# ============================================================
# Display Target
# ============================================================

print("\n")

print("="*70)
print("First Five Target Values")
print("="*70)

display(y.head())

# ============================================================
# Final Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Final Dataset Summary")
print("="*70)

print("Total Samples  :", len(df))

print("Total Features :", X.shape[1])

print("Target Classes :", target_encoder.classes_)

print("\nDataset is Ready for XGBoost Training.")

print("="*70)

# ============================================================
#  Model Building & Training
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\n")

print("="*70)
print("Train Test Split")
print("="*70)

print("Training Features :", X_train.shape)

print("Testing Features  :", X_test.shape)

print("Training Labels   :", y_train.shape)

print("Testing Labels    :", y_test.shape)

# ============================================================
# Create XGBoost Model
# ============================================================

xgb_model = XGBClassifier(

    n_estimators=50,

    max_depth=3,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="multi:softprob",

    num_class=3,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42,

    n_jobs=-1

)

# ============================================================
# Train Model
# ============================================================

xgb_model.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)
print("XGBoost Model Trained Successfully")
print("="*70)

# ============================================================
# Make Predictions
# ============================================================

predictions = xgb_model.predict(

    X_test

)

# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("Accuracy")
print("="*70)

print("Accuracy :", round(accuracy,4))

# ============================================================
# Precision
# ============================================================

precision = precision_score(

    y_test,

    predictions,

    average="weighted"

)

print("\n")

print("="*70)
print("Precision")
print("="*70)

print("Precision :", round(precision,4))

# ============================================================
# Recall
# ============================================================

recall = recall_score(

    y_test,

    predictions,

    average="weighted"

)

print("\n")

print("="*70)
print("Recall")
print("="*70)

print("Recall :", round(recall,4))

# ============================================================
# F1 Score
# ============================================================

f1 = f1_score(

    y_test,

    predictions,

    average="weighted"

)

print("\n")

print("="*70)
print("F1 Score")
print("="*70)

print("F1 Score :", round(f1,4))

# ============================================================
# Cross Validation
# ============================================================

cv_scores = cross_val_score(

    xgb_model,

    X,

    y,

    cv=2,

    scoring="accuracy",

    n_jobs=-1

)

print("\n")

print("="*70)
print("Cross Validation")
print("="*70)

print("Scores :", cv_scores)

print("Average Accuracy :", round(cv_scores.mean(),4))

# ============================================================
# Hyperparameter Tuning
# ============================================================

parameters = {

    "n_estimators":[50],

    "max_depth":[3],

    "learning_rate":[0.1]

}

grid_search = GridSearchCV(

    estimator=xgb_model,

    param_grid=parameters,

    cv=2,

    scoring="accuracy",

    n_jobs=-1

)

grid_search.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)
print("Best Parameters")
print("="*70)

print(grid_search.best_params_)

# ============================================================
# Best Model
# ============================================================

best_model = grid_search.best_estimator_

best_predictions = best_model.predict(

    X_test

)

# ============================================================
# Evaluate Best Model
# ============================================================

best_accuracy = accuracy_score(

    y_test,

    best_predictions

)

best_precision = precision_score(

    y_test,

    best_predictions,

    average="weighted"

)

best_recall = recall_score(

    y_test,

    best_predictions,

    average="weighted"

)

best_f1 = f1_score(

    y_test,

    best_predictions,

    average="weighted"

)

print("\n")

print("="*70)
print("Best XGBoost Model Performance")
print("="*70)

print("Accuracy  :", round(best_accuracy,4))

print("Precision :", round(best_precision,4))

print("Recall    :", round(best_recall,4))

print("F1 Score  :", round(best_f1,4))

# ============================================================
# Feature Importance
# ============================================================

feature_importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":best_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)
print("Feature Importance")
print("="*70)

print(feature_importance)

# ============================================================
# Feature Importance Graph
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=feature_importance,

    x="Importance",

    y="Feature"

)

plt.title("XGBoost Feature Importance")

plt.show()

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("XGBoost Model Summary")
print("="*70)

print(f"Training Samples : {X_train.shape[0]}")

print(f"Testing Samples  : {X_test.shape[0]}")

print(f"Accuracy         : {best_accuracy:.4f}")

print(f"Precision        : {best_precision:.4f}")

print(f"Recall           : {best_recall:.4f}")

print(f"F1 Score         : {best_f1:.4f}")

print(f"Cross Validation : {cv_scores.mean():.4f}")

print("\n")

print("Model Status : Ready for Evaluation")

print("="*70)

# ============================================================
#  Model Evaluation & Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from joblib import dump
from joblib import load

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_test,

    best_predictions

)

print("\n")

print("="*70)
print("Confusion Matrix")
print("="*70)

print(cm)

plt.figure(figsize=(7,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# Classification Report
# ============================================================

print("\n")

print("="*70)
print("Classification Report")
print("="*70)

print(

    classification_report(

        y_test,

        best_predictions

    )

)

# ============================================================
# Predict New Customer
# ============================================================

new_customer = X.iloc[[0]].copy()

prediction = best_model.predict(

    new_customer

)

print("\n")

print("="*70)
print("Prediction for New Customer")
print("="*70)

predicted_class = target_encoder.inverse_transform(

    prediction.astype(int)

)

print("Predicted Credit Score :", predicted_class[0])

# ============================================================
# Prediction Probability
# ============================================================

probability = best_model.predict_proba(

    new_customer

)

print("\n")

print("="*70)
print("Prediction Probability")
print("="*70)

print(probability)

# ============================================================
# Save Model
# ============================================================

dump(

    best_model,

    "credit_scoring_xgboost_model.joblib"

)

print("\n")

print("="*70)
print("Model Saved Successfully")
print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = load(

    "credit_scoring_xgboost_model.joblib"

)

print("\n")

print("="*70)
print("Saved Model Loaded Successfully")
print("="*70)

# ============================================================
# Predict Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    new_customer

)

loaded_prediction = target_encoder.inverse_transform(

    loaded_prediction.astype(int)

)

print("\n")

print("="*70)
print("Prediction Using Loaded Model")
print("="*70)

print("Predicted Credit Score :", loaded_prediction[0])

# ============================================================
# Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual" : target_encoder.inverse_transform(

        y_test.iloc[:10].values.astype(int)

    ),

    "Predicted" : target_encoder.inverse_transform(

        best_predictions[:10].astype(int)

    )

})

print("\n")

print("="*70)
print("Actual vs Predicted")
print("="*70)

print(comparison)

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature" : X.columns,

    "Importance" : best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)
print("Feature Importance")
print("="*70)

print(importance)

# ============================================================
# Feature Importance Graph
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=importance,

    x="Importance",

    y="Feature"

)

plt.title("XGBoost Feature Importance")

plt.show()

# ============================================================
# Top 10 Important Features
# ============================================================

top10 = importance.head(10)

plt.figure(figsize=(10,6))

sns.barplot(

    data=top10,

    x="Importance",

    y="Feature"

)

plt.title("Top 10 Important Features")

plt.show()

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Credit Scoring using XGBoost Classifier")
print("="*70)

print(f"Training Samples : {X_train.shape[0]}")

print(f"Testing Samples  : {X_test.shape[0]}")

print(f"Total Features   : {X.shape[1]}")

print(f"Accuracy         : {best_accuracy:.4f}")

print(f"Precision        : {best_precision:.4f}")

print(f"Recall           : {best_recall:.4f}")

print(f"F1 Score         : {best_f1:.4f}")

print(f"Cross Validation : {cv_scores.mean():.4f}")

print("\n")

print("Algorithm Used : XGBoost Classifier")

print("Model Status   : Ready for Deployment")

print("="*70)

print("\nProject Completed Successfully!")
