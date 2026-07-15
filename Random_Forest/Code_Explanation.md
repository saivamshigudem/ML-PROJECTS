# ============================================================
# Fraud Detection using Random Forest Classifier
# Data Loading, EDA & Data Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inside Jupyter Notebook
%matplotlib inline

# Better plot style

sns.set_style("whitegrid")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("synthetic_fraud_dataset.csv")

# ============================================================
# Basic Dataset Exploration
# ============================================================

print("="*70)
print("First 5 Records")
print("="*70)

print(df.head())

print("\n")

print("="*70)
print("Last 5 Records")
print("="*70)

print(df.tail())

print("\n")

print("="*70)
print("Dataset Shape")
print("="*70)

print(df.shape)

print("\n")

print("="*70)
print("Column Names")
print("="*70)

print(df.columns)

print("\n")

print("="*70)
print("Dataset Information")
print("="*70)

df.info()

print("\n")

print("="*70)
print("Data Types")
print("="*70)

print(df.dtypes)

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("="*70)
print("Missing Values")
print("="*70)

print(df.isnull().sum())

# ============================================================
# Duplicate Records
# ============================================================

print("\n")

print("="*70)
print("Duplicate Records")
print("="*70)

print(df.duplicated().sum())

# Remove duplicate rows

df = df.drop_duplicates()

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*70)
print("Statistical Summary")
print("="*70)

print(df.describe())

# ============================================================
# Separate Numerical & Categorical Columns
# ============================================================

from pandas.api.types import (

    is_string_dtype,

    is_integer_dtype,

    is_float_dtype

)

categorical_columns = []

numerical_columns = []

for col in df.columns:

    if is_string_dtype(df[col]):

        categorical_columns.append(col)

    elif is_integer_dtype(df[col]) or is_float_dtype(df[col]):

        numerical_columns.append(col)

print("\n")

print("="*70)
print("Categorical Columns")
print("="*70)

print(categorical_columns)

print("\n")

print("="*70)
print("Numerical Columns")
print("="*70)

print(numerical_columns)

# ============================================================
# Explore Categorical Columns
# ============================================================

for col in categorical_columns:

    print("\n")

    print("="*70)
    print(col.upper())
    print("="*70)

    print(df[col].value_counts())

# ============================================================
# Handle Missing Values
# ============================================================

# Fill categorical columns using Mode

for col in categorical_columns:

    df[col] = df[col].fillna(

        df[col].mode()[0]

    )

# Fill numerical columns using Median

for col in numerical_columns:

    df[col] = df[col].fillna(

        df[col].median()

    )

print("\n")

print("="*70)
print("Missing Values After Imputation")
print("="*70)

print(df.isnull().sum())

# ============================================================
# Feature Engineering
# ============================================================

# Convert Timestamp into datetime

df["Timestamp"] = pd.to_datetime(

    df["Timestamp"]

)

# Extract useful features

df["Year"] = df["Timestamp"].dt.year

df["Month"] = df["Timestamp"].dt.month

df["Day"] = df["Timestamp"].dt.day

df["Hour"] = df["Timestamp"].dt.hour

df["Day_of_Week"] = df["Timestamp"].dt.dayofweek

# Remove original timestamp

df = df.drop(

    columns=["Timestamp"]

)

print("\n")

print("="*70)
print("Dataset After Feature Engineering")
print("="*70)

print(df.head())

# ============================================================
# Fraud Distribution
# ============================================================

print("\n")

print("="*70)
print("Fraud Distribution")
print("="*70)

print(df["Fraud_Label"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    data=df,

    x="Fraud_Label"

)

plt.title("Fraud vs Genuine Transactions")

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

plt.figure(figsize=(12,8))

sns.heatmap(

    df.select_dtypes(include=np.number).corr(),

    annot=True,

    cmap="coolwarm"

)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# Histograms
# ============================================================

hist_columns = [

    "Transaction_Amount",

    "Account_Balance",

    "Risk_Score",

    "Transaction_Distance"

]

for col in hist_columns:

    plt.figure(figsize=(7,4))

    sns.histplot(

        df[col],

        kde=True

    )

    plt.title(col)

    plt.show()

# ============================================================
# Boxplots
# ============================================================

box_columns = [

    "Transaction_Amount",

    "Account_Balance",

    "Transaction_Distance"

]

for col in box_columns:

    plt.figure(figsize=(7,2))

    sns.boxplot(

        x=df[col]

    )

    plt.title(col)

    plt.show()

# ============================================================
# Fraud Label vs Numerical Features
# ============================================================

features = [

    "Transaction_Amount",

    "Risk_Score",

    "Account_Balance",

    "Transaction_Distance"

]

for feature in features:

    plt.figure(figsize=(7,5))

    sns.boxplot(

        data=df,

        x="Fraud_Label",

        y=feature

    )

    plt.title(f"{feature} vs Fraud Label")

    plt.show()

# ============================================================
# Countplots
# ============================================================

categorical_features = [

    "Transaction_Type",

    "Device_Type",

    "Location",

    "Merchant_Category",

    "Card_Type",

    "Authentication_Method"

]

for feature in categorical_features:

    plt.figure(figsize=(8,4))

    sns.countplot(

        data=df,

        x=feature,

        hue="Fraud_Label"

    )

    plt.xticks(rotation=30)

    plt.title(feature)

    plt.show()

# ============================================================
# Prepare Features & Target
# ============================================================

# Remove identifier columns

X = df.drop(

    columns=[

        "Transaction_ID",

        "User_ID",

        "Fraud_Label"

    ]

)

# Target variable

y = df["Fraud_Label"]

print("\n")

print("="*70)
print("Input Features")
print("="*70)

print(X.head())

print("\n")

print("="*70)
print("Target Variable")
print("="*70)

print(y.head())

print("\n")

print("Input Shape :", X.shape)

print("Target Shape:", y.shape)

# ============================================================
# Data Encoding & Model Building
# ============================================================

# ============================================================
# One-Hot Encoding
# ============================================================

# Convert categorical columns into numerical columns

categorical_columns = [

    "Transaction_Type",

    "Device_Type",

    "Location",

    "Merchant_Category",

    "Card_Type",

    "Authentication_Method"

]

df = pd.get_dummies(

    df,

    columns=categorical_columns,

    drop_first=True,

    dtype=int

)

# ============================================================
# Prepare Features & Target
# ============================================================

X = df.drop(

    columns=[

        "Transaction_ID",

        "User_ID",

        "Fraud_Label"

    ]

)

y = df["Fraud_Label"]

print("="*70)
print("Encoded Dataset")
print("="*70)

print(X.head())

# ============================================================
# Train-Test Split
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape")

print(X_train.shape)

print(y_train.shape)

print("\nTesting Shape")

print(X_test.shape)

print(y_test.shape)

# ============================================================
# Train Random Forest Classifier
# ============================================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(

    random_state=42

)

rf_model.fit(

    X_train,

    y_train

)

print("\nRandom Forest Model Trained Successfully!")

# ============================================================
# Predictions
# ============================================================

y_pred = rf_model.predict(

    X_test

)

y_prob = rf_model.predict_proba(

    X_test

)

# ============================================================
# Model Evaluation
# ============================================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score

)

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

print("\n")

print("="*70)
print("Default Random Forest Performance")
print("="*70)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

# ============================================================
# Hyperparameter Tuning
# ============================================================

from sklearn.model_selection import GridSearchCV

parameters = {

    "n_estimators":[

        100,

        200,

        300

    ],

    "criterion":[

        "gini",

        "entropy"

    ],

    "max_depth":[

        None,

        10,

        20,

        30

    ],

    "min_samples_split":[

        2,

        5,

        10

    ]

}

grid = GridSearchCV(

    estimator=RandomForestClassifier(

        random_state=42

    ),

    param_grid=parameters,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)
print("Best Parameters")
print("="*70)

print(grid.best_params_)

# ============================================================
# Best Random Forest Model
# ============================================================

best_model = grid.best_estimator_

best_predictions = best_model.predict(

    X_test

)

# ============================================================
# Best Model Evaluation
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
print("Best Random Forest Performance")
print("="*70)

print(f"Accuracy  : {best_accuracy:.4f}")

print(f"Precision : {best_precision:.4f}")

print(f"Recall    : {best_recall:.4f}")

print(f"F1 Score  : {best_f1:.4f}")

# ============================================================
# Cross Validation
# ============================================================

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(

    best_model,

    X,

    y,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

print("\n")

print("="*70)
print("Cross Validation")
print("="*70)

print(cv_scores)

print("\nAverage Accuracy")

print(cv_scores.mean())

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)
print("Top 20 Important Features")
print("="*70)

print(importance.head(20))

# ============================================================
# Model Evaluation & Deployment
# ============================================================

# ============================================================
# Confusion Matrix
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(

    y_test,

    best_predictions

)

print("="*70)
print("Confusion Matrix")
print("="*70)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=["Genuine","Fraud"],

    yticklabels=["Genuine","Fraud"]

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# Classification Report
# ============================================================

from sklearn.metrics import classification_report

print("\n")

print("="*70)
print("Classification Report")
print("="*70)

print(

    classification_report(

        y_test,

        best_predictions,

        target_names=["Genuine","Fraud"]

    )

)

# ============================================================
# ROC Curve
# ============================================================

from sklearn.metrics import roc_curve, roc_auc_score

fraud_probability = best_model.predict_proba(

    X_test

)[:,1]

fpr, tpr, threshold = roc_curve(

    y_test,

    fraud_probability

)

auc_score = roc_auc_score(

    y_test,

    fraud_probability

)

plt.figure(figsize=(7,6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {auc_score:.4f}"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    color="red"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("\nROC AUC Score :", auc_score)

# ============================================================
# Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": best_predictions

})

print("\n")

print("="*70)
print("Sample Predictions")
print("="*70)

print(comparison.head(20))

# ============================================================
# Feature Importance Plot
# ============================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

plt.figure(figsize=(10,8))

sns.barplot(

    data=importance.head(15),

    x="Importance",

    y="Feature"

)

plt.title("Top 15 Important Features")

plt.show()

# ============================================================
# Predict New Transaction
# ============================================================

new_transaction = pd.DataFrame({

    "Transaction_Amount":[250.50],

    "Account_Balance":[50000],

    "IP_Address_Flag":[0],

    "Daily_Transaction_Count":[5],

    "Avg_Transaction_Amount_7d":[180],

    "Failed_Transaction_Count_7d":[1],

    "Card_Age":[120],

    "Transaction_Distance":[200],

    "Risk_Score":[0.35],

    "Is_Weekend":[0],

    "Year":[2023],

    "Month":[10],

    "Day":[15],

    "Hour":[14],

    "Day_of_Week":[6]

})

# Add remaining encoded columns

for col in X.columns:

    if col not in new_transaction.columns:

        new_transaction[col] = 0

# Arrange columns exactly like training data

new_transaction = new_transaction[X.columns]

prediction = best_model.predict(

    new_transaction

)

probability = best_model.predict_proba(

    new_transaction

)

print("\n")

print("="*70)
print("Fraud Prediction")
print("="*70)

if prediction[0] == 1:

    print("Prediction : FRAUD TRANSACTION")

else:

    print("Prediction : GENUINE TRANSACTION")

print("\nPrediction Probability")

print(probability)

# ============================================================
# Save Model
# ============================================================

from joblib import dump

dump(

    best_model,

    "fraud_detection_random_forest.joblib"

)

print("\nModel Saved Successfully!")

# ============================================================
# Load Model
# ============================================================

from joblib import load

loaded_model = load(

    "fraud_detection_random_forest.joblib"

)

print("Saved Model Loaded Successfully!")

# ============================================================
# Predict Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    new_transaction

)

print("\nPrediction Using Loaded Model")

if loaded_prediction[0] == 1:

    print("Fraud Transaction")

else:

    print("Genuine Transaction")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Fraud Detection using Random Forest")
print("="*70)

print(f"Training Samples        : {X_train.shape[0]}")

print(f"Testing Samples         : {X_test.shape[0]}")

print(f"Total Features          : {X.shape[1]}")

print(f"\nAccuracy               : {best_accuracy:.4f}")

print(f"Precision              : {best_precision:.4f}")

print(f"Recall                 : {best_recall:.4f}")

print(f"F1 Score               : {best_f1:.4f}")

print(f"ROC AUC Score          : {auc_score:.4f}")

print(f"Cross Validation       : {cv_scores.mean():.4f}")

print("\nModel Status : Ready for Deployment")

print("="*70)

print("\nProject Completed Successfully!")
