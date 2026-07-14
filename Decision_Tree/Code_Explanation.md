# ============================================================
# Loan Approval Prediction using Decision Tree Classifier
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

# Load training dataset

df = pd.read_csv("train.csv")

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
# Explore Categorical Features
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

# Fill categorical missing values using Mode

categorical_missing = [

    "Gender",

    "Married",

    "Dependents",

    "Self_Employed"

]

for col in categorical_missing:

    df[col] = df[col].fillna(

        df[col].mode()[0]

    )

# Fill numerical missing values using Median

numerical_missing = [

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History"

]

for col in numerical_missing:

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

# Create Total Income Feature

df["Total_Income"] = (

    df["ApplicantIncome"]

    +

    df["CoapplicantIncome"]

)

print("\n")

print("="*70)
print("Dataset with Total Income")
print("="*70)

print(df.head())

# ============================================================
# Target Distribution
# ============================================================

print("\n")

print("="*70)
print("Loan Status Distribution")
print("="*70)

print(df["Loan_Status"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    data=df,

    x="Loan_Status"

)

plt.title("Loan Approval Distribution")

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

plt.figure(figsize=(8,6))

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

    "ApplicantIncome",

    "CoapplicantIncome",

    "LoanAmount",

    "Total_Income"

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

    "ApplicantIncome",

    "LoanAmount",

    "Total_Income"

]

for col in box_columns:

    plt.figure(figsize=(7,2))

    sns.boxplot(

        x=df[col]

    )

    plt.title(col)

    plt.show()

# ============================================================
# Loan Status vs Numerical Features
# ============================================================

features = [

    "ApplicantIncome",

    "LoanAmount",

    "Total_Income"

]

for feature in features:

    plt.figure(figsize=(7,5))

    sns.boxplot(

        data=df,

        x="Loan_Status",

        y=feature

    )

    plt.title(f"{feature} vs Loan Status")

    plt.show()

# ============================================================
# Countplots
# ============================================================

categorical_features = [

    "Gender",

    "Married",

    "Education",

    "Self_Employed",

    "Property_Area"

]

for feature in categorical_features:

    plt.figure(figsize=(7,4))

    sns.countplot(

        data=df,

        x=feature,

        hue="Loan_Status"

    )

    plt.xticks(rotation=25)

    plt.title(feature)

    plt.show()

# ============================================================
# Feature Selection
# ============================================================

# Loan_ID is only an identifier

X = df.drop(

    columns=[

        "Loan_ID",

        "Loan_Status"

    ]

)

# Target Variable

y = df["Loan_Status"]

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
#  Data Encoding & Model Building
# ============================================================

# ============================================================
# Convert Dependents Column
# ============================================================

# Replace "3+" with 3

df["Dependents"] = df["Dependents"].replace("3+", "3")

# Convert to integer

df["Dependents"] = df["Dependents"].astype(int)

# ============================================================
# One-Hot Encoding
# ============================================================

# Convert categorical columns into numerical columns

categorical_columns = [

    "Gender",

    "Married",

    "Education",

    "Self_Employed",

    "Property_Area"

]

df = pd.get_dummies(

    df,

    columns=categorical_columns,

    drop_first=True,

    dtype=int

)

# ============================================================
# Encode Target Variable
# ============================================================

# Y -> 1
# N -> 0

df["Loan_Status"] = df["Loan_Status"].map({

    "Y":1,

    "N":0

})

# ============================================================
# Prepare Features & Target
# ============================================================

X = df.drop(

    columns=[

        "Loan_ID",

        "Loan_Status"

    ]

)

y = df["Loan_Status"]

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
# Train Decision Tree (Default)
# ============================================================

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(

    random_state=42

)

dt_model.fit(

    X_train,

    y_train

)

print("\nDecision Tree Model Trained Successfully!")

# ============================================================
# Predictions
# ============================================================

y_pred = dt_model.predict(

    X_test

)

y_prob = dt_model.predict_proba(

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
print("Default Decision Tree Performance")
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

    "criterion":[

        "gini",

        "entropy"

    ],

    "max_depth":[

        2,

        3,

        4,

        5,

        6,

        8,

        10

    ],

    "min_samples_split":[

        2,

        5,

        10,

        20

    ]

}

grid = GridSearchCV(

    estimator=DecisionTreeClassifier(

        random_state=42

    ),

    param_grid=parameters,

    cv=5,

    scoring="accuracy"

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
# Best Decision Tree Model
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
print("Best Decision Tree Performance")
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

    scoring="accuracy"

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

    "Feature":X.columns,

    "Importance":best_model.feature_importances_

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

    xticklabels=["Rejected","Approved"],

    yticklabels=["Rejected","Approved"]

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

        target_names=["Rejected","Approved"]

    )

)

# ============================================================
# ROC Curve
# ============================================================

from sklearn.metrics import roc_curve, roc_auc_score

probability = best_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(

    y_test,

    probability

)

auc_score = roc_auc_score(

    y_test,

    probability

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
# Decision Tree Visualization
# ============================================================

from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))

plot_tree(

    best_model,

    feature_names=X.columns,

    class_names=["Rejected","Approved"],

    filled=True,

    rounded=True,

    fontsize=9

)

plt.title("Decision Tree")

plt.show()

# ============================================================
# Feature Importance Graph
# ============================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

plt.figure(figsize=(10,6))

sns.barplot(

    data=importance,

    x="Importance",

    y="Feature"

)

plt.title("Feature Importance")

plt.show()

# ============================================================
# Predict New Applicant
# ============================================================

new_applicant = pd.DataFrame({

    "Dependents":[1],

    "ApplicantIncome":[5000],

    "CoapplicantIncome":[1500],

    "LoanAmount":[150],

    "Loan_Amount_Term":[360],

    "Credit_History":[1],

    "Total_Income":[6500],

    "Gender_Male":[1],

    "Married_Yes":[1],

    "Education_Not Graduate":[0],

    "Self_Employed_Yes":[0],

    "Property_Area_Semiurban":[1],

    "Property_Area_Urban":[0]

})

prediction = best_model.predict(

    new_applicant

)

probability = best_model.predict_proba(

    new_applicant

)

print("\n")

print("="*70)
print("Loan Prediction")
print("="*70)

if prediction[0] == 1:

    print("Loan Status : APPROVED")

else:

    print("Loan Status : REJECTED")

print("\nPrediction Probability")

print(probability)

# ============================================================
# Save Model
# ============================================================

from joblib import dump

dump(

    best_model,

    "loan_decision_tree_model.joblib"

)

print("\nModel Saved Successfully!")

# ============================================================
# Load Model
# ============================================================

from joblib import load

loaded_model = load(

    "loan_decision_tree_model.joblib"

)

print("Saved Model Loaded Successfully!")

# ============================================================
# Predict Using Loaded Model
# ============================================================

prediction = loaded_model.predict(

    new_applicant

)

print("\nPrediction Using Loaded Model")

if prediction[0] == 1:

    print("Loan Approved")

else:

    print("Loan Rejected")

# ============================================================
# Predict Test Dataset
# ============================================================

test_df = pd.read_csv("test.csv")

# Save Loan_ID separately

loan_ids = test_df["Loan_ID"]

# Fill Missing Values

categorical_missing = [

    "Gender",

    "Married",

    "Dependents",

    "Self_Employed"

]

for col in categorical_missing:

    test_df[col] = test_df[col].fillna(

        test_df[col].mode()[0]

    )

numerical_missing = [

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History"

]

for col in numerical_missing:

    test_df[col] = test_df[col].fillna(

        test_df[col].median()

    )

# Convert Dependents

test_df["Dependents"] = test_df["Dependents"].replace(

    "3+",

    "3"

).astype(int)

# Create Total Income

test_df["Total_Income"] = (

    test_df["ApplicantIncome"]

    +

    test_df["CoapplicantIncome"]

)

# One-Hot Encoding

test_df = pd.get_dummies(

    test_df,

    columns=[

        "Gender",

        "Married",

        "Education",

        "Self_Employed",

        "Property_Area"

    ],

    drop_first=True,

    dtype=int

)

# Remove Loan_ID

test_features = test_df.drop(

    columns=["Loan_ID"]

)

# Match training columns

test_features = test_features.reindex(

    columns=X.columns,

    fill_value=0

)

predictions = loaded_model.predict(

    test_features

)

submission = pd.DataFrame({

    "Loan_ID":loan_ids,

    "Loan_Status":np.where(

        predictions==1,

        "Y",

        "N"

    )

})

print("\n")

print("="*70)
print("Test Dataset Predictions")
print("="*70)

print(submission.head())

submission.to_csv(

    "Loan_Predictions.csv",

    index=False

)

print("\nPrediction File Saved Successfully!")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Loan Approval Prediction using Decision Tree")
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

