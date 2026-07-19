# ============================================================
# Customer Analytics using LightGBM Classifier
#  Data Loading & Exploratory Data Analysis
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

df = pd.read_csv("marketing_campaign.csv", delimiter="\t")

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

numerical_columns = df.select_dtypes(

    include=["int64","float64"]

).columns

print("\n")

print("="*70)
print("Numerical Columns")
print("="*70)

print(numerical_columns.tolist())

# ============================================================
# Categorical Columns
# ============================================================

categorical_columns = df.select_dtypes(

    include=["object"]

).columns

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
# Target Variable Distribution
# ============================================================

print("\n")

print("="*70)
print("Target Variable Distribution")
print("="*70)

print(df["Response"].value_counts())

# ============================================================
# Target Percentage
# ============================================================

print("\n")

print("="*70)
print("Target Percentage")
print("="*70)

print(round(df["Response"].value_counts(normalize=True)*100,2))

# ============================================================
# Target Count Plot
# ============================================================

plt.figure(figsize=(6,4))

sns.countplot(

    x="Response",

    data=df

)

plt.title("Campaign Response Distribution")

plt.show()

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Histograms")
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

plt.figure(figsize=(16,12))

sns.heatmap(

    df.corr(numeric_only=True),

    cmap="coolwarm",

    annot=True,

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
# Campaign Response Percentage
# ============================================================

response_percentage = round(

    df["Response"].value_counts(normalize=True) * 100,

    2

)

print("\n")

print("="*70)
print("Campaign Response Percentage")
print("="*70)

print(response_percentage)
# ============================================================
# Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.preprocessing import LabelEncoder

# ============================================================
# Remove Unnecessary Columns
# ============================================================

columns_to_drop = [

    "ID"

]

existing_columns = [

    column

    for column in columns_to_drop

    if column in df.columns

]

df.drop(

    columns=existing_columns,

    inplace=True

)

print("\n")

print("="*70)
print("Unnecessary Columns Removed")
print("="*70)

print(df.columns.tolist())

# ============================================================
# Check Missing Values
# ============================================================

print("\n")

print("="*70)
print("Missing Values Before Treatment")
print("="*70)

print(df.isnull().sum())

# ============================================================
# Fill Missing Values
# ============================================================

import pandas as pd
from pandas.api.types import is_numeric_dtype

for column in df.columns:
    if is_numeric_dtype(df[column]):
        # Fill numeric columns with median
        df[column].fillna(df[column].median(), inplace=True)
    else:
        # Fill non-numeric (object, string, category) with mode
        df[column].fillna(df[column].mode()[0], inplace=True)

print("\n")
print("="*70)
print("Missing Values After Treatment")
print("="*70)
print(df.isnull().sum())

# ============================================================
# Encode Target Variable
# ============================================================

target_encoder = LabelEncoder()

df["Response"] = target_encoder.fit_transform(

    df["Response"]

)

print("\n")

print("="*70)
print("Target Variable Encoded")
print("="*70)

print(target_encoder.classes_)

# ============================================================
# Encode Categorical Features
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
# Feature Engineering
# ============================================================

current_year = 2026

df["Age"] = current_year - df["Year_Birth"]

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Total_Spending"] = (

    df["MntWines"]

    + df["MntFruits"]

    + df["MntMeatProducts"]

    + df["MntFishProducts"]

    + df["MntSweetProducts"]

    + df["MntGoldProds"]

)

df["Total_Purchases"] = (

    df["NumWebPurchases"]

    + df["NumCatalogPurchases"]

    + df["NumStorePurchases"]

)

print("\n")

print("="*70)
print("Feature Engineering Completed")
print("="*70)

# ============================================================
# Feature Matrix and Target Variable
# ============================================================

X = df.drop(

    "Response",

    axis=1

)

y = df["Response"]

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

print("\nDataset is Ready for LightGBM Training.")

print("="*70)

# ============================================================
# Model Building & Training
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

from lightgbm import LGBMClassifier

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
# Create LightGBM Model
# ============================================================

lgbm_model = LGBMClassifier(

    n_estimators=50,

    learning_rate=0.1,

    max_depth=5,

    num_leaves=31,

    random_state=42,

    n_jobs=-1,

    verbosity=-1

)

# ============================================================
# Train Model
# ============================================================

lgbm_model.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)
print("LightGBM Model Trained Successfully")
print("="*70)

# ============================================================
# Make Predictions
# ============================================================

predictions = lgbm_model.predict(

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

    predictions

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

    predictions

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

    predictions

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

    lgbm_model,

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

    "learning_rate":[0.1],

    "max_depth":[5]

}

grid_search = GridSearchCV(

    estimator=lgbm_model,

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

    best_predictions

)

best_recall = recall_score(

    y_test,

    best_predictions

)

best_f1 = f1_score(

    y_test,

    best_predictions

)

print("\n")

print("="*70)
print("Best LightGBM Model Performance")
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

plt.title("LightGBM Feature Importance")

plt.show()

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("LightGBM Model Summary")
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

import joblib

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

# ============================================================
# Confusion Matrix Heatmap
# ============================================================

plt.figure(figsize=(6,5))

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

prediction_label = target_encoder.inverse_transform(

    prediction

)

print("\n")

print("="*70)
print("Prediction for New Customer")
print("="*70)

print("Predicted Class :", prediction_label[0])

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

joblib.dump(

    best_model,

    "customer_analytics_lightgbm_model.joblib"

)

print("\n")

print("="*70)
print("Model Saved Successfully")
print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = joblib.load(

    "customer_analytics_lightgbm_model.joblib"

)

print("\n")

print("="*70)
print("Saved Model Loaded Successfully")
print("="*70)

# ============================================================
# Prediction Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    new_customer

)

loaded_prediction_label = target_encoder.inverse_transform(

    loaded_prediction

)

print("\n")

print("="*70)
print("Prediction Using Loaded Model")
print("="*70)

print("Predicted Class :", loaded_prediction_label[0])

# ============================================================
# Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual" : target_encoder.inverse_transform(

        y_test.iloc[:20]

    ),

    "Predicted" : target_encoder.inverse_transform(

        best_predictions[:20]

    )

})

print("\n")

print("="*70)
print("Actual vs Predicted")
print("="*70)

display(comparison)

# ============================================================
# Top 10 Important Features
# ============================================================

top_features = feature_importance.head(10)

print("\n")

print("="*70)
print("Top 10 Important Features")
print("="*70)

display(top_features)

# ============================================================
# Top 10 Feature Importance Graph
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=top_features,

    x="Importance",

    y="Feature"

)

plt.title("Top 10 Feature Importance")

plt.show()

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Customer Analytics using LightGBM")
print("="*70)

print("Algorithm Used        : LightGBM Classifier")

print("Training Samples      :", X_train.shape[0])

print("Testing Samples       :", X_test.shape[0])

print("Total Features        :", X.shape[1])

print("Accuracy              :", round(best_accuracy,4))

print("Precision             :", round(best_precision,4))

print("Recall                :", round(best_recall,4))

print("F1 Score              :", round(best_f1,4))

print("Cross Validation      :", round(cv_scores.mean(),4))

print("Model File            : customer_analytics_lightgbm_model.joblib")

print("Deployment Status     : Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Customer Analytics using LightGBM Completed Successfully!")

print("="*70)

