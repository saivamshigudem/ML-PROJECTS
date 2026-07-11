# ============================================================
# Customer Churn Prediction using Logistic Regression
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inside notebook
%matplotlib inline

# Better plot style
sns.set_style("whitegrid")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("customer_churn_dataset.csv")

# ============================================================
# Basic Dataset Exploration
# ============================================================

print("="*60)
print("First 5 Records")
print("="*60)

print(df.head())

print("\n")

print("="*60)
print("Last 5 Records")
print("="*60)

print(df.tail())

print("\n")

print("="*60)
print("Dataset Shape")
print("="*60)

print(df.shape)

print("\n")

print("="*60)
print("Column Names")
print("="*60)

print(df.columns)

print("\n")

print("="*60)
print("Dataset Information")
print("="*60)

df.info()

print("\n")

print("="*60)
print("Data Types")
print("="*60)

print(df.dtypes)

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("="*60)
print("Missing Values")
print("="*60)

print(df.isnull().sum())

# ============================================================
# Duplicate Records
# ============================================================

print("\n")

print("="*60)
print("Duplicate Records")
print("="*60)

print(df.duplicated().sum())

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*60)
print("Statistical Summary")
print("="*60)

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

print("="*60)
print("Categorical Columns")
print("="*60)

print(categorical_columns)

print("\n")

print("="*60)
print("Numerical Columns")
print("="*60)

print(numerical_columns)

# ============================================================
# Explore Categorical Features
# ============================================================

for col in categorical_columns:

    print("\n")

    print("="*60)
    print(col.upper())
    print("="*60)

    print(df[col].value_counts())

# ============================================================
# Target Distribution
# ============================================================

print("\n")

print("="*60)
print("Target Distribution")
print("="*60)

print(df["Churn"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    data=df,

    x="Churn"

)

plt.title("Customer Churn Distribution")

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))

sns.heatmap(

    correlation,

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

)

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# Histograms
# ============================================================

numerical_features = [

    "Age",

    "Tenure",

    "Usage Frequency",

    "Support Calls",

    "Payment Delay",

    "Total Spend",

    "Last Interaction"

]

for feature in numerical_features:

    plt.figure(figsize=(7,4))

    sns.histplot(

        df[feature],

        kde=True

    )

    plt.title(f"Distribution of {feature}")

    plt.show()

# ============================================================
# Boxplots
# ============================================================

for feature in numerical_features:

    plt.figure(figsize=(7,2))

    sns.boxplot(

        x=df[feature]

    )

    plt.title(feature)

    plt.show()

# ============================================================
# Relationship with Target
# ============================================================

for feature in numerical_features:

    plt.figure(figsize=(7,5))

    sns.boxplot(

        data=df,

        x="Churn",

        y=feature

    )

    plt.title(f"{feature} vs Churn")

    plt.show()

# ============================================================
# Countplots for Categorical Features
# ============================================================

categorical_features = [

    "Gender",

    "Subscription Type",

    "Contract Length"

]

for feature in categorical_features:

    plt.figure(figsize=(7,4))

    sns.countplot(

        data=df,

        x=feature,

        hue="Churn"

    )

    plt.xticks(rotation=30)

    plt.title(feature)

    plt.show()

# ============================================================
# Pairplot
# ============================================================

sns.pairplot(

    df[

        [

            "Age",

            "Tenure",

            "Usage Frequency",

            "Total Spend",

            "Churn"

        ]

    ],

    hue="Churn"

)

plt.show()

# ============================================================
# Feature Selection
# ============================================================

# CustomerID is only an identifier

X = df.drop(

    columns=[

        "CustomerID",

        "Churn"

    ]

)

# Target Variable

y = df["Churn"]

print("\n")

print("="*60)
print("Input Features")
print("="*60)

print(X.head())

print("\n")

print("="*60)
print("Target")
print("="*60)

print(y.head())

print("\n")

print("Input Shape :", X.shape)

print("Target Shape:", y.shape)

print("\n")


# ============================================================
# One-Hot Encoding
# ============================================================

# Convert categorical columns into numerical columns
# drop_first=True avoids the Dummy Variable Trap

categorical_features = [

    "Gender",

    "Subscription Type",

    "Contract Length"

]

df_encoded = pd.get_dummies(

    df,

    columns=categorical_features,

    drop_first=True,

    dtype=int

)

print("="*60)
print("Encoded Dataset")
print("="*60)

print(df_encoded.head())

print("\n")

print("Dataset Shape :", df_encoded.shape)

# ============================================================
# Features & Target
# ============================================================

X = df_encoded.drop(

    columns=[

        "CustomerID",

        "Churn"

    ]

)

y = df_encoded["Churn"]

print("\n")

print("="*60)
print("Feature Matrix")
print("="*60)

print(X.head())

print("\n")

print("="*60)
print("Target Variable")
print("="*60)

print(y.head())

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

print("\n")

print("="*60)
print("Training Data")
print("="*60)

print(X_train.shape)

print(y_train.shape)

print("\n")

print("="*60)
print("Testing Data")
print("="*60)

print(X_test.shape)

print(y_test.shape)

# ============================================================
# Feature Scaling
# ============================================================

# Logistic Regression generally performs better when
# numerical features are on a similar scale.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

# ============================================================
# Logistic Regression Model
# ============================================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

# Train the model

model.fit(

    X_train_scaled,

    y_train

)

print("\nModel Training Completed Successfully!")

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(

    X_test_scaled

)

# Probability Predictions

y_prob = model.predict_proba(

    X_test_scaled

)

print("\n")

print("="*60)
print("Predicted Classes")
print("="*60)

print(y_pred[:10])

print("\n")

print("="*60)
print("Prediction Probabilities")
print("="*60)

print(y_prob[:10])

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

print("="*60)
print("Model Performance")
print("="*60)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

# ============================================================
# Cross Validation
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(

        max_iter=1000,

        random_state=42

    ))

])

cv_scores = cross_val_score(

    pipeline,

    X,

    y,

    cv=5,

    scoring="accuracy"

)

print("\n")

print("="*60)
print("Cross Validation")
print("="*60)

print("Fold Scores")

print(cv_scores)

print("\nAverage Accuracy")

print(cv_scores.mean())

# ============================================================
# Feature Importance
# ============================================================

coefficients = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": model.coef_[0]

})

coefficients["Absolute Coefficient"] = abs(

    coefficients["Coefficient"]

)

coefficients = coefficients.sort_values(

    by="Absolute Coefficient",

    ascending=False

)

print("\n")

print("="*60)
print("Feature Importance")
print("="*60)

print(coefficients)

# ============================================================
# Top 10 Important Features
# ============================================================

top10 = coefficients.head(10)

plt.figure(figsize=(10,6))

sns.barplot(

    data=top10,

    x="Absolute Coefficient",

    y="Feature"

)

plt.title("Top 10 Important Features")

plt.xlabel("Absolute Coefficient")

plt.ylabel("Feature")

plt.show()

# ============================================================
# Model Summary
# ============================================================

print("\n")

print("="*60)
print("Logistic Regression Summary")
print("="*60)

print(f"Training Samples       : {len(X_train)}")

print(f"Testing Samples        : {len(X_test)}")

print(f"Total Features         : {X_train.shape[1]}")

print(f"\nAccuracy              : {accuracy:.4f}")

print(f"Precision             : {precision:.4f}")

print(f"Recall                : {recall:.4f}")

print(f"F1 Score              : {f1:.4f}")

print(f"Cross Validation      : {cv_scores.mean():.4f}")


# ============================================================
# Confusion Matrix
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(

    y_test,

    y_pred

)

print("="*60)
print("Confusion Matrix")
print("="*60)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=["No Churn", "Churn"],

    yticklabels=["No Churn", "Churn"]

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

print("="*60)
print("Classification Report")
print("="*60)

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ============================================================
# ROC Curve
# ============================================================

from sklearn.metrics import (

    roc_curve,

    roc_auc_score

)

# Probability of positive class

y_prob_positive = y_prob[:,1]

# Calculate ROC values

fpr, tpr, thresholds = roc_curve(

    y_test,

    y_prob_positive

)

auc_score = roc_auc_score(

    y_test,

    y_prob_positive

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

print("\n")

print("="*60)
print("ROC AUC Score")
print("="*60)

print(auc_score)

# ============================================================
# Actual vs Predicted Comparison
# ============================================================

comparison = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": y_pred

})

print("\n")

print("="*60)
print("Sample Predictions")
print("="*60)

print(comparison.head(15))

# ============================================================
# Predict New Customer
# ============================================================

# IMPORTANT:
# Column names and order must match X_train

new_customer = pd.DataFrame({

    "Age":[35],

    "Tenure":[24],

    "Usage Frequency":[18],

    "Support Calls":[2],

    "Payment Delay":[5],

    "Total Spend":[650],

    "Last Interaction":[10],

    "Gender_Male":[1],

    "Subscription Type_Premium":[0],

    "Subscription Type_Standard":[1],

    "Contract Length_Monthly":[1],

    "Contract Length_Quarterly":[0]

})

# Scale input

new_customer_scaled = scaler.transform(

    new_customer

)

# Predict class

prediction = model.predict(

    new_customer_scaled

)

# Predict probabilities

prediction_probability = model.predict_proba(

    new_customer_scaled

)

print("\n")

print("="*60)
print("Prediction for New Customer")
print("="*60)

print("Predicted Class :", prediction[0])

if prediction[0] == 0:

    print("Customer is likely to Stay.")

else:

    print("Customer is likely to Churn.")

print("\nPrediction Probability")

print(prediction_probability)

# ============================================================
# Save Model & Scaler
# ============================================================

from joblib import dump

dump(

    model,

    "logistic_regression_model.joblib"

)

dump(

    scaler,

    "standard_scaler.joblib"

)

print("\nModel Saved Successfully!")

# ============================================================
# Load Model & Scaler
# ============================================================

from joblib import load

loaded_model = load(

    "logistic_regression_model.joblib"

)

loaded_scaler = load(

    "standard_scaler.joblib"

)

print("Saved Model Loaded Successfully!")

# ============================================================
# Prediction Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    loaded_scaler.transform(new_customer)

)

print("\nPrediction Using Loaded Model")

print(loaded_prediction)

# ============================================================
# Final Project Summary
# ============================================================

print("\n")
print("="*70)
print("Customer Churn Prediction using Logistic Regression")
print("="*70)

print(f"Training Samples           : {len(X_train)}")

print(f"Testing Samples            : {len(X_test)}")

print(f"Total Features             : {X_train.shape[1]}")

print(f"\nAccuracy                  : {accuracy:.4f}")

print(f"Precision                 : {precision:.4f}")

print(f"Recall                    : {recall:.4f}")

print(f"F1 Score                  : {f1:.4f}")

print(f"ROC AUC Score             : {auc_score:.4f}")

print(f"Cross Validation Accuracy : {cv_scores.mean():.4f}")

print("\nModel Status : Ready for Deployment")

print("="*70)

print("\nProject Completed Successfully!")
