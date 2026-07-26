# ============================================================
# Fraud Detection using Isolation Forest
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

df = pd.read_csv("creditcard.csv")

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

display(df.isnull().sum())

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
# Target Distribution
# ============================================================

print("\n")

print("="*70)
print("Fraud Distribution")
print("="*70)

display(df["Class"].value_counts())

plt.figure(figsize=(8,5))

sns.countplot(

    x="Class",

    data=df

)

plt.title("Fraud vs Normal Transactions")

plt.xlabel("Class")

plt.ylabel("Count")

plt.show()

# ============================================================
# Fraud Percentage
# ============================================================

fraud_percentage = (

    df["Class"].value_counts(normalize=True)

    *100

).round(4)

print("\n")

print("="*70)
print("Fraud Percentage")
print("="*70)

display(fraud_percentage)

# ============================================================
# Transaction Amount Distribution
# ============================================================

print("\n")

print("="*70)
print("Transaction Amount Distribution")
print("="*70)

plt.figure(figsize=(10,5))

sns.histplot(

    df["Amount"],

    bins=50,

    kde=True

)

plt.title("Transaction Amount Distribution")

plt.xlabel("Amount")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# Transaction Time Distribution
# ============================================================

print("\n")

print("="*70)
print("Transaction Time Distribution")
print("="*70)

plt.figure(figsize=(10,5))

sns.histplot(

    df["Time"],

    bins=50,

    kde=True

)

plt.title("Transaction Time Distribution")

plt.xlabel("Time")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

print("\n")

print("="*70)
print("Correlation Matrix")
print("="*70)

plt.figure(figsize=(15,12))

sns.heatmap(

    df.corr(),

    cmap="coolwarm",

    center=0

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

selected_features = [

    "Time",

    "Amount",

    "V1",

    "V2",

    "V3",

    "V4"

]

for column in selected_features:

    plt.figure(figsize=(8,4))

    sns.boxplot(

        x=df[column]

    )

    plt.title(column)

    plt.show()

# ============================================================
# Histograms of Important Features
# ============================================================

print("\n")

print("="*70)
print("Feature Distributions")
print("="*70)

for column in selected_features:

    plt.figure(figsize=(8,4))

    sns.histplot(

        df[column],

        bins=40,

        kde=True

    )

    plt.title(f"{column} Distribution")

    plt.show()

# ============================================================
# Correlation with Target
# ============================================================

print("\n")

print("="*70)
print("Correlation with Fraud Class")
print("="*70)

target_corr = df.corr()["Class"].sort_values(

    ascending=False

)

display(target_corr)

# ============================================================
# Top Fraud Transactions
# ============================================================

print("\n")

print("="*70)
print("Sample Fraud Transactions")
print("="*70)

display(

    df[df["Class"]==1].head(10)

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Dataset Summary")
print("="*70)

print("Total Records          :", df.shape[0])

print("Total Features         :", df.shape[1]-1)

print("Fraud Transactions     :", df["Class"].sum())

print("Normal Transactions    :", len(df)-df["Class"].sum())

print("Fraud Percentage (%)   :",

      round(

          df["Class"].mean()*100,

          4

      ))

print("="*70)
# ============================================================
#  Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.preprocessing import StandardScaler

# ============================================================
# Check Missing Values Before Treatment
# ============================================================

print("\n")

print("="*70)
print("Missing Values Before Treatment")
print("="*70)

display(df.isnull().sum())

# ============================================================
# Handle Missing Values
# ============================================================

for column in df.columns:

    if df[column].dtype == "object":

        df[column].fillna(

            df[column].mode()[0],

            inplace=True

        )

    else:

        df[column].fillna(

            df[column].median(),

            inplace=True

        )

print("\n")

print("="*70)
print("Missing Values After Treatment")
print("="*70)

display(df.isnull().sum())

# ============================================================
# Separate Features and Target
# ============================================================

X = df.drop(

    "Class",

    axis=1

)

y = df["Class"]

print("\n")

print("="*70)
print("Feature Matrix and Target Variable")
print("="*70)

print("Feature Shape :", X.shape)

print("Target Shape  :", y.shape)

# ============================================================
# Display Feature Matrix
# ============================================================

print("\n")

print("="*70)
print("Feature Matrix")
print("="*70)

display(X.head())

# ============================================================
# Display Target Variable
# ============================================================

print("\n")

print("="*70)
print("Target Variable")
print("="*70)

display(y.head())

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(

    X

)

print("\n")

print("="*70)
print("Feature Scaling Completed")
print("="*70)

print("Scaled Data Shape :", X_scaled.shape)

# ============================================================
# Convert Scaled Data into DataFrame
# ============================================================

X_scaled_df = pd.DataFrame(

    X_scaled,

    columns=X.columns

)

print("\n")

print("="*70)
print("Scaled Feature Matrix")
print("="*70)

display(X_scaled_df.head())

# ============================================================
# Verify Mean
# ============================================================

print("\n")

print("="*70)
print("Mean of Scaled Features")
print("="*70)

display(

    X_scaled_df.mean().round(2)

)

# ============================================================
# Verify Standard Deviation
# ============================================================

print("\n")

print("="*70)
print("Standard Deviation of Scaled Features")
print("="*70)

display(

    X_scaled_df.std().round(2)

)

# ============================================================
# Class Distribution
# ============================================================

print("\n")

print("="*70)
print("Target Class Distribution")
print("="*70)

class_distribution = pd.DataFrame({

    "Count": y.value_counts(),

    "Percentage (%)": (y.value_counts(normalize=True)*100).round(4)

})

display(class_distribution)

# ============================================================
# Feature Statistics
# ============================================================

feature_statistics = pd.DataFrame({

    "Mean": X.mean(),

    "Median": X.median(),

    "Standard Deviation": X.std(),

    "Minimum": X.min(),

    "Maximum": X.max()

})

print("\n")

print("="*70)
print("Feature Statistics")
print("="*70)

display(feature_statistics.head(10))

# ============================================================
# Correlation Matrix of Scaled Features
# ============================================================

print("\n")

print("="*70)
print("Correlation Matrix (Scaled Features)")
print("="*70)

plt.figure(figsize=(15,12))

sns.heatmap(

    X_scaled_df.corr(),

    cmap="coolwarm",

    center=0

)

plt.title("Correlation Matrix (Scaled Features)")

plt.show()

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Dataset Summary")
print("="*70)

print("Total Records          :", X.shape[0])

print("Total Features         :", X.shape[1])

print("Fraud Transactions     :", y.sum())

print("Normal Transactions    :", len(y)-y.sum())

print("Fraud Percentage (%)   :", round(y.mean()*100,4))

print("Scaled Dataset Shape   :", X_scaled.shape)

print("\nDataset is Ready for Isolation Forest Training.")

print("="*70)

# ============================================================
#  Model Building & Evaluation
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.ensemble import IsolationForest

from sklearn.metrics import (

    confusion_matrix,

    classification_report,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)

# ============================================================
# Build Isolation Forest Model
# ============================================================

model = IsolationForest(

    n_estimators=100,

    contamination=0.0017,

    random_state=42,

    n_jobs=-1

)

print("\n")

print("="*70)

print("Isolation Forest Model Created Successfully")

print("="*70)

# ============================================================
# Train Model
# ============================================================

model.fit(

    X_scaled

)

print("\n")

print("="*70)

print("Model Training Completed Successfully")

print("="*70)

# ============================================================
# Predict Anomalies
# ============================================================

predictions = model.predict(

    X_scaled

)

print("\n")

print("="*70)

print("Predictions Generated Successfully")

print("="*70)

# ============================================================
# Convert Predictions
# ============================================================

predicted_class = np.where(

    predictions == -1,

    1,

    0

)

# ============================================================
# Anomaly Score
# ============================================================

anomaly_score = -model.decision_function(

    X_scaled

)

df["Anomaly_Score"] = anomaly_score

df["Predicted_Class"] = predicted_class

# ============================================================
# Prediction Summary
# ============================================================

print("\n")

print("="*70)

print("Prediction Summary")

print("="*70)

summary = pd.DataFrame({

    "Count": pd.Series(predicted_class).value_counts()

})

display(summary)

# ============================================================
# Fraud Percentage Detected
# ============================================================

fraud_percentage = (

    predicted_class.sum()

    /

    len(predicted_class)

)*100

print("\n")

print("="*70)

print("Detected Fraud Percentage")

print("="*70)

print(

    round(fraud_percentage,4),

    "%"

)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y,

    predicted_class

)

print("\n")

print("="*70)

print("Confusion Matrix")

print("="*70)

display(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=["Normal","Fraud"],

    yticklabels=["Normal","Fraud"]

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

        y,

        predicted_class

    )

)

# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(

    y,

    predicted_class

)

precision = precision_score(

    y,

    predicted_class,

    zero_division=0

)

recall = recall_score(

    y,

    predicted_class,

    zero_division=0

)

f1 = f1_score(

    y,

    predicted_class,

    zero_division=0

)

roc_auc = roc_auc_score(

    y,

    predicted_class

)

metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1,

        roc_auc

    ]

})

print("\n")

print("="*70)

print("Evaluation Metrics")

print("="*70)

display(metrics)

# ============================================================
# Top 10 Highest Anomaly Scores
# ============================================================

print("\n")

print("="*70)

print("Top 10 Highest Anomaly Scores")

print("="*70)

display(

    df.sort_values(

        by="Anomaly_Score",

        ascending=False

    ).head(10)

)

# ============================================================
# Anomaly Score Distribution
# ============================================================

plt.figure(figsize=(10,5))

sns.histplot(

    df["Anomaly_Score"],

    bins=50,

    kde=True

)

plt.title("Anomaly Score Distribution")

plt.xlabel("Anomaly Score")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# Amount vs Anomaly Score
# ============================================================

plt.figure(figsize=(10,6))

sns.scatterplot(

    data=df,

    x="Amount",

    y="Anomaly_Score",

    hue="Predicted_Class",

    alpha=0.6

)

plt.title("Amount vs Anomaly Score")

plt.show()

# ============================================================
# Actual vs Predicted Fraud
# ============================================================

comparison = pd.DataFrame({

    "Actual": y,

    "Predicted": predicted_class

})

print("\n")

print("="*70)

print("Actual vs Predicted")

print("="*70)

display(

    comparison.head(20)

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Isolation Forest isolates anomalies by randomly partitioning the feature space.")

print("- Fraudulent transactions usually receive higher anomaly scores.")

print("- The contamination parameter controls the expected proportion of anomalies.")

print("- Labels were used only for evaluation, not during training.")

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)

print("Isolation Forest Summary")

print("="*70)

print("Accuracy        :", round(accuracy,4))

print("Precision       :", round(precision,4))

print("Recall          :", round(recall,4))

print("F1 Score        :", round(f1,4))

print("ROC-AUC         :", round(roc_auc,4))

print("Detected Fraud  :", predicted_class.sum())

print("Normal Records  :", len(predicted_class)-predicted_class.sum())

print("="*70)

# ============================================================
# Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save Scaler
# ============================================================

joblib.dump(

    scaler,

    "fraud_detection_scaler.joblib"

)

# ============================================================
# Save Isolation Forest Model
# ============================================================

joblib.dump(

    model,

    "isolation_forest_model.joblib"

)

print("\n")

print("="*70)

print("Scaler and Isolation Forest Model Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Models
# ============================================================

loaded_scaler = joblib.load(

    "fraud_detection_scaler.joblib"

)

loaded_model = joblib.load(

    "isolation_forest_model.joblib"

)

print("\n")

print("="*70)

print("Saved Models Loaded Successfully")

print("="*70)

# ============================================================
# Predict New Transaction
# ============================================================

new_transaction = X.iloc[[0]]

scaled_transaction = loaded_scaler.transform(

    new_transaction

)

prediction = loaded_model.predict(

    scaled_transaction

)

anomaly_score = -loaded_model.decision_function(

    scaled_transaction

)

predicted_label = np.where(

    prediction == -1,

    "Fraud",

    "Normal"

)

print("\n")

print("="*70)

print("New Transaction Prediction")

print("="*70)

print("Prediction      :", predicted_label[0])

print("Anomaly Score   :", round(anomaly_score[0],6))

# ============================================================
# Prediction Confidence
# ============================================================

confidence = pd.DataFrame({

    "Prediction":[predicted_label[0]],

    "Anomaly Score":[round(anomaly_score[0],6)]

})

print("\n")

print("="*70)

print("Prediction Summary")

print("="*70)

display(confidence)

# ============================================================
# Export Prediction Results
# ============================================================

prediction_results = df.copy()

prediction_results.to_csv(

    "Fraud_Detection_Output.csv",

    index=False

)

print("\n")

print("="*70)

print("Prediction Results Exported Successfully")

print("="*70)

# ============================================================
# Top 20 Suspicious Transactions
# ============================================================

top_fraud = prediction_results.sort_values(

    by="Anomaly_Score",

    ascending=False

).head(20)

print("\n")

print("="*70)

print("Top 20 Suspicious Transactions")

print("="*70)

display(top_fraud)

top_fraud.to_csv(

    "Top_20_Suspicious_Transactions.csv",

    index=False

)

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Estimators",

        "Contamination",

        "Random State"

    ],

    "Value":[

        "Isolation Forest",

        model.n_estimators,

        model.contamination,

        model.random_state

    ]

})

print("\n")

print("="*70)

print("Model Information")

print("="*70)

display(model_information)

# ============================================================
# Deployment Summary
# ============================================================

deployment_summary = pd.DataFrame({

    "Item":[

        "Scaler File",

        "Model File",

        "Prediction File",

        "Top Fraud File"

    ],

    "Output":[

        "fraud_detection_scaler.joblib",

        "isolation_forest_model.joblib",

        "Fraud_Detection_Output.csv",

        "Top_20_Suspicious_Transactions.csv"

    ]

})

print("\n")

print("="*70)

print("Deployment Files")

print("="*70)

display(deployment_summary)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Isolation Forest identifies rare and unusual transactions without requiring labeled training data.")

print("- Transactions with higher anomaly scores should be prioritized for fraud investigation.")

print("- The model can serve as an early warning system before manual verification.")

print("- High-risk transactions can trigger additional authentication such as OTP or biometric verification.")

print("- The model should be retrained periodically as transaction behavior evolves over time.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Fraud Detection using Isolation Forest")

print("="*70)

print("Dataset                 : Credit Card Fraud Detection")

print("Algorithm               : Isolation Forest")

print("Training Records        :", X.shape[0])

print("Features Used           :", X.shape[1])

print("Fraud Detected          :", predicted_class.sum())

print("Normal Transactions     :", len(predicted_class)-predicted_class.sum())

print("Accuracy                :", round(accuracy,4))

print("Precision               :", round(precision,4))

print("Recall                  :", round(recall,4))

print("F1 Score                :", round(f1,4))

print("ROC-AUC                 :", round(roc_auc,4))

print("Scaler Saved            : fraud_detection_scaler.joblib")

print("Model Saved             : isolation_forest_model.joblib")

print("Prediction File         : Fraud_Detection_Output.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Fraud Detection using Isolation Forest Completed Successfully!")

print("="*70)

