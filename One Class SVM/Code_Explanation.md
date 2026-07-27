# ============================================================
# Intrusion Detection using One-Class SVM
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
# Define Column Names
# ============================================================

columns = [

    "duration",

    "protocol_type",

    "service",

    "flag",

    "src_bytes",

    "dst_bytes",

    "land",

    "wrong_fragment",

    "urgent",

    "hot",

    "num_failed_logins",

    "logged_in",

    "num_compromised",

    "root_shell",

    "su_attempted",

    "num_root",

    "num_file_creations",

    "num_shells",

    "num_access_files",

    "num_outbound_cmds",

    "is_host_login",

    "is_guest_login",

    "count",

    "srv_count",

    "serror_rate",

    "srv_serror_rate",

    "rerror_rate",

    "srv_rerror_rate",

    "same_srv_rate",

    "diff_srv_rate",

    "srv_diff_host_rate",

    "dst_host_count",

    "dst_host_srv_count",

    "dst_host_same_srv_rate",

    "dst_host_diff_srv_rate",

    "dst_host_same_src_port_rate",

    "dst_host_srv_diff_host_rate",

    "dst_host_serror_rate",

    "dst_host_srv_serror_rate",

    "dst_host_rerror_rate",

    "dst_host_srv_rerror_rate",

    "label"

]

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(

    "KDDTrain+.txt",

    names=columns

)

print("\n")

print("="*70)

print("Dataset Loaded Successfully")

print("="*70)

# ============================================================
# First Five Records
# ============================================================

print("\n")

print("="*70)

print("First Five Records")

print("="*70)

display(df.head())

# ============================================================
# Last Five Records
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

df.drop_duplicates(

    inplace=True

)

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

    include="object"

).columns

print("\n")

print("="*70)

print("Categorical Columns")

print("="*70)

print(categorical_columns.tolist())

# ============================================================
# Target Distribution
# ============================================================

print("\n")

print("="*70)

print("Attack Distribution")

print("="*70)

display(

    df["label"].value_counts()

)

plt.figure(figsize=(12,6))

sns.countplot(

    y="label",

    data=df,

    order=df["label"].value_counts().index

)

plt.title("Attack Distribution")

plt.show()

# ============================================================
# Protocol Distribution
# ============================================================

print("\n")

print("="*70)

print("Protocol Distribution")

print("="*70)

display(

    df["protocol_type"].value_counts()

)

plt.figure(figsize=(7,5))

sns.countplot(

    x="protocol_type",

    data=df

)

plt.title("Protocol Distribution")

plt.show()

# ============================================================
# Service Distribution
# ============================================================

print("\n")

print("="*70)

print("Top 15 Services")

print("="*70)

display(

    df["service"].value_counts().head(15)

)

plt.figure(figsize=(12,6))

sns.countplot(

    y="service",

    data=df,

    order=df["service"].value_counts().head(15).index

)

plt.title("Top 15 Services")

plt.show()

# ============================================================
# Flag Distribution
# ============================================================

print("\n")

print("="*70)

print("Flag Distribution")

print("="*70)

display(

    df["flag"].value_counts()

)

plt.figure(figsize=(10,5))

sns.countplot(

    x="flag",

    data=df,

    order=df["flag"].value_counts().index

)

plt.title("Flag Distribution")

plt.xticks(rotation=45)

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

    df[numerical_columns].corr(),

    cmap="coolwarm",

    center=0

)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# Histograms
# ============================================================

selected_features = [

    "duration",

    "src_bytes",

    "dst_bytes",

    "count",

    "srv_count"

]

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
# Boxplots
# ============================================================

print("\n")

print("="*70)

print("Boxplots")

print("="*70)

for column in selected_features:

    plt.figure(figsize=(8,4))

    sns.boxplot(

        x=df[column]

    )

    plt.title(column)

    plt.show()

# ============================================================
# Sample Attack Records
# ============================================================

print("\n")

print("="*70)

print("Sample Attack Records")

print("="*70)

display(

    df[df["label"]!="normal"].head(10)

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print("Total Records          :", df.shape[0])

print("Total Features         :", df.shape[1]-2)

print("Numerical Features     :", len(numerical_columns))

print("Categorical Features   :", len(categorical_columns))

print("Attack Categories      :", df["label"].nunique())

print("Most Common Protocol   :", df["protocol_type"].mode()[0])

print("Most Common Service    :", df["service"].mode()[0])

print("="*70)

# ============================================================
#  Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

# ============================================================
# Remove Duplicate Columns (If Any)
# ============================================================

print("\n")

print("="*70)

print("Checking Duplicate Columns")

print("="*70)

duplicate_columns = df.columns[df.columns.duplicated()]

print("Duplicate Columns :", duplicate_columns.tolist())

df = df.loc[:, ~df.columns.duplicated()]

print("\n")

print("="*70)

print("Duplicate Columns Removed Successfully")

print("="*70)

print("Current Shape :", df.shape)

# ============================================================
# Drop Difficulty Column
# ============================================================

if "difficulty" in df.columns:

    df.drop(

        "difficulty",

        axis=1,

        inplace=True

    )

print("\n")

print("="*70)

print("Difficulty Column Removed")

print("="*70)

print("Current Shape :", df.shape)

# ============================================================
# Missing Value Check
# ============================================================

print("\n")

print("="*70)

print("Missing Values Before Treatment")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Handle Missing Values
# ============================================================

from pandas.api.types import is_numeric_dtype

for column in df.columns:
    if is_numeric_dtype(df[column]):
        # Numeric → fill with median
        df[column].fillna(df[column].median(), inplace=True)
    else:
        # Non-numeric → fill with mode
        if not df[column].mode().empty:
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna("missing", inplace=True)
        # Ensure categorical columns are strings
        df[column] = df[column].astype(str)

print("\n")
print("="*70)
print("Missing Values After Treatment")
print("="*70)
display(df.isnull().sum())
# ============================================================
# Encode Categorical Features
# ============================================================

categorical_columns = [

    "protocol_type",

    "service",

    "flag"

]

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(

        df[column]

    )

    label_encoders[column] = encoder

print("\n")

print("="*70)

print("Categorical Features Encoded Successfully")

print("="*70)

display(

    df[categorical_columns].head()

)

# ============================================================
# Convert Target into Binary
# ============================================================

df["label"] = np.where(

    df["label"] == "normal",

    0,

    1

)

print("\n")

print("="*70)

print("Target Converted Successfully")

print("="*70)

display(

    df["label"].value_counts()

)

# ============================================================
# Separate Features & Target
# ============================================================

X = df.drop(

    "label",

    axis=1

)

y = df["label"]

print("\n")

print("="*70)

print("Feature Matrix and Target Variable")

print("="*70)

print("Feature Shape :", X.shape)

print("Target Shape  :", y.shape)

# ============================================================
# Separate Normal and Attack Data
# ============================================================

X_train = X[

    y == 0

]

X_test = X.copy()

y_test = y.copy()

print("\n")

print("="*70)

print("Training and Testing Data")

print("="*70)

print("Training Records :", X_train.shape[0])

print("Testing Records  :", X_test.shape[0])

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(

    X_train

)

X_test_scaled = scaler.transform(

    X_test

)

print("\n")

print("="*70)

print("Feature Scaling Completed")

print("="*70)

print("Training Shape :", X_train_scaled.shape)

print("Testing Shape  :", X_test_scaled.shape)

# ============================================================
# Convert Scaled DataFrames
# ============================================================

X_train_scaled_df = pd.DataFrame(

    X_train_scaled,

    columns=X.columns

)

X_test_scaled_df = pd.DataFrame(

    X_test_scaled,

    columns=X.columns

)

print("\n")

print("="*70)

print("Scaled Training Dataset")

print("="*70)

display(

    X_train_scaled_df.head()

)

# ============================================================
# Verify Mean
# ============================================================

print("\n")

print("="*70)

print("Mean of Scaled Features")

print("="*70)

display(

    X_train_scaled_df.mean().round(2)

)

# ============================================================
# Verify Standard Deviation
# ============================================================

print("\n")

print("="*70)

print("Standard Deviation of Scaled Features")

print("="*70)

display(

    X_train_scaled_df.std().round(2)

)

# ============================================================
# Target Distribution
# ============================================================

class_distribution = pd.DataFrame({

    "Count": y.value_counts(),

    "Percentage (%)":

    (

        y.value_counts(

            normalize=True

        ) * 100

    ).round(2)

})

print("\n")

print("="*70)

print("Target Distribution")

print("="*70)

display(

    class_distribution

)

# ============================================================
# Correlation Matrix
# ============================================================

print("\n")

print("="*70)

print("Correlation Matrix")

print("="*70)

plt.figure(

    figsize=(16,12)

)

sns.heatmap(

    X_train_scaled_df.corr(),

    cmap="coolwarm",

    center=0

)

plt.title(

    "Correlation Matrix"

)

plt.show()

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print("Training Samples          :", X_train.shape[0])

print("Testing Samples           :", X_test.shape[0])

print("Number of Features        :", X.shape[1])

print("Normal Samples            :", (y==0).sum())

print("Attack Samples            :", (y==1).sum())

print("Training Data             : Only Normal Traffic")

print("Testing Data              : Normal + Attack Traffic")

print("\nDataset Ready for One-Class SVM")

print("="*70)

# ============================================================
#  Model Building & Evaluation
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.svm import OneClassSVM

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
# Build One-Class SVM Model
# ============================================================

model = OneClassSVM(

    kernel="rbf",

    gamma="scale",

    nu=0.02

)

print("\n")

print("="*70)

print("One-Class SVM Model Created Successfully")

print("="*70)

# ============================================================
# Train Model
# ============================================================

model.fit(

    X_train_scaled

)

print("\n")

print("="*70)

print("Model Training Completed Successfully")

print("="*70)

# ============================================================
# Predict Intrusions
# ============================================================

predictions = model.predict(

    X_test_scaled

)

print("\n")

print("="*70)

print("Predictions Generated Successfully")

print("="*70)

# ============================================================
# Convert Predictions
# ============================================================

predicted_label = np.where(

    predictions == -1,

    1,

    0

)

# ============================================================
# Decision Function Scores
# ============================================================

decision_scores = -model.decision_function(

    X_test_scaled

)

df["Decision_Score"] = decision_scores

df["Predicted_Label"] = predicted_label

# ============================================================
# Prediction Summary
# ============================================================

print("\n")

print("="*70)

print("Prediction Summary")

print("="*70)

prediction_summary = pd.DataFrame({

    "Count": pd.Series(

        predicted_label

    ).value_counts()

})

display(

    prediction_summary

)

# ============================================================
# Intrusion Detection Percentage
# ============================================================

intrusion_percentage = (

    predicted_label.sum()

    /

    len(predicted_label)

)*100

print("\n")

print("="*70)

print("Detected Intrusion Percentage")

print("="*70)

print(

    round(

        intrusion_percentage,

        4

    ),

    "%"

)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_test,

    predicted_label

)

print("\n")

print("="*70)

print("Confusion Matrix")

print("="*70)

display(

    cm

)

plt.figure(

    figsize=(6,5)

)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[

        "Normal",

        "Intrusion"

    ],

    yticklabels=[

        "Normal",

        "Intrusion"

    ]

)

plt.xlabel(

    "Predicted"

)

plt.ylabel(

    "Actual"

)

plt.title(

    "Confusion Matrix"

)

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

        predicted_label

    )

)

# ============================================================
# Evaluation Metrics
# ============================================================

accuracy = accuracy_score(

    y_test,

    predicted_label

)

precision = precision_score(

    y_test,

    predicted_label,

    zero_division=0

)

recall = recall_score(

    y_test,

    predicted_label,

    zero_division=0

)

f1 = f1_score(

    y_test,

    predicted_label,

    zero_division=0

)

roc_auc = roc_auc_score(

    y_test,

    predicted_label

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

display(

    metrics

)

# ============================================================
# Top Suspicious Connections
# ============================================================

print("\n")

print("="*70)

print("Top 10 Suspicious Connections")

print("="*70)

display(

    df.sort_values(

        by="Decision_Score",

        ascending=False

    ).head(10)

)

# ============================================================
# Decision Score Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    df["Decision_Score"],

    bins=50,

    kde=True

)

plt.title(

    "Decision Score Distribution"

)

plt.xlabel(

    "Decision Score"

)

plt.ylabel(

    "Frequency"

)

plt.show()

# ============================================================
# Decision Score vs Count
# ============================================================

plt.figure(

    figsize=(10,6)

)

sns.scatterplot(

    x=df["count"],

    y=df["Decision_Score"],

    hue=df["Predicted_Label"],

    alpha=0.6

)

plt.title(

    "Count vs Decision Score"

)

plt.xlabel(

    "Connection Count"

)

plt.ylabel(

    "Decision Score"

)

plt.show()

# ============================================================
# Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual": y_test,

    "Predicted": predicted_label

})

print("\n")

print("="*70)

print("Actual vs Predicted")

print("="*70)

display(

    comparison.head(20)

)

# ============================================================
# Top Decision Scores
# ============================================================

top_scores = df.sort_values(

    by="Decision_Score",

    ascending=False

).head(20)

print("\n")

print("="*70)

print("Top 20 Suspicious Connections")

print("="*70)

display(

    top_scores

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- One-Class SVM learns only normal network behavior.")

print("- Connections outside the learned boundary are classified as intrusions.")

print("- Decision scores indicate how abnormal a connection is.")

print("- Higher decision scores represent more suspicious network activity.")

print("- Labels were used only for evaluating the model.")

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)

print("One-Class SVM Summary")

print("="*70)

print("Accuracy           :", round(accuracy,4))

print("Precision          :", round(precision,4))

print("Recall             :", round(recall,4))

print("F1 Score           :", round(f1,4))

print("ROC-AUC            :", round(roc_auc,4))

print("Detected Intrusions:", predicted_label.sum())

print("Normal Connections :", len(predicted_label)-predicted_label.sum())

print("="*70)

# ============================================================
#  Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save StandardScaler
# ============================================================

joblib.dump(

    scaler,

    "intrusion_scaler.joblib"

)

# ============================================================
# Save One-Class SVM Model
# ============================================================

joblib.dump(

    model,

    "one_class_svm_model.joblib"

)

# ============================================================
# Save Label Encoders
# ============================================================

joblib.dump(

    label_encoders,

    "label_encoders.joblib"

)

print("\n")

print("="*70)

print("Models Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Models
# ============================================================

loaded_scaler = joblib.load(

    "intrusion_scaler.joblib"

)

loaded_model = joblib.load(

    "one_class_svm_model.joblib"

)

loaded_encoders = joblib.load(

    "label_encoders.joblib"

)

print("\n")

print("="*70)

print("Saved Models Loaded Successfully")

print("="*70)

# ============================================================
# Predict New Network Connection
# ============================================================

new_connection = X.iloc[[0]]

scaled_connection = loaded_scaler.transform(

    new_connection

)

prediction = loaded_model.predict(

    scaled_connection

)

decision_score = -loaded_model.decision_function(

    scaled_connection

)

predicted_label = np.where(

    prediction == -1,

    "Intrusion",

    "Normal"

)

print("\n")

print("="*70)

print("New Connection Prediction")

print("="*70)

print("Prediction      :", predicted_label[0])

print("Decision Score  :", round(decision_score[0],6))

# ============================================================
# Prediction Summary
# ============================================================

prediction_summary = pd.DataFrame({

    "Prediction":[

        predicted_label[0]

    ],

    "Decision Score":[

        round(

            decision_score[0],

            6

        )

    ]

})

print("\n")

print("="*70)

print("Prediction Summary")

print("="*70)

display(

    prediction_summary

)

# ============================================================
# Export Detection Results
# ============================================================

output = df.copy()

output.to_csv(

    "Intrusion_Detection_Output.csv",

    index=False

)

print("\n")

print("="*70)

print("Detection Results Exported Successfully")

print("="*70)

# ============================================================
# Export Top Suspicious Connections
# ============================================================

top_intrusions = output.sort_values(

    by="Decision_Score",

    ascending=False

).head(20)

top_intrusions.to_csv(

    "Top_20_Suspicious_Connections.csv",

    index=False

)

print("\n")

print("="*70)

print("Top 20 Suspicious Connections")

print("="*70)

display(

    top_intrusions

)

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Kernel",

        "Gamma",

        "Nu"

    ],

    "Value":[

        "One-Class SVM",

        model.kernel,

        model.gamma,

        model.nu

    ]

})

print("\n")

print("="*70)

print("Model Information")

print("="*70)

display(

    model_information

)

# ============================================================
# Deployment Files
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "Scaler",

        "Model",

        "Label Encoders",

        "Detection Output",

        "Top Suspicious Connections"

    ],

    "Saved As":[

        "intrusion_scaler.joblib",

        "one_class_svm_model.joblib",

        "label_encoders.joblib",

        "Intrusion_Detection_Output.csv",

        "Top_20_Suspicious_Connections.csv"

    ]

})

print("\n")

print("="*70)

print("Deployment Files")

print("="*70)

display(

    deployment_summary

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- One-Class SVM learns only normal network behavior.")

print("- Connections outside the learned boundary are flagged as intrusions.")

print("- Decision scores help prioritize suspicious network events.")

print("- High-score connections should be investigated immediately.")

print("- Retraining the model periodically helps adapt to evolving attack patterns.")

detected_intrusions = np.sum(predicted_label == -1)

print("Detected Intrusions     :", detected_intrusions)

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Intrusion Detection using One-Class SVM")

print("="*70)

print("Dataset                 : NSL-KDD")

print("Algorithm               : One-Class SVM")

print("Kernel                  :", model.kernel)

print("Gamma                   :", model.gamma)

print("Nu                      :", model.nu)

print("Training Samples        :", X_train.shape[0])

print("Testing Samples         :", X_test.shape[0])

print("Number of Features      :", X.shape[1])

if isinstance(predicted_label, np.ndarray):
    # Adjust depending on your label encoding
    detected_intrusions = np.sum(predicted_label == -1)   # if -1 marks intrusions
    # detected_intrusions = np.sum(predicted_label == "attack")  # if string labels
    print("Detected Intrusions     :", detected_intrusions)
else:
    print("Detected Intrusions     : N/A")
    
print("Accuracy                :", round(accuracy,4))

print("Precision               :", round(precision,4))

print("Recall                  :", round(recall,4))

print("F1 Score                :", round(f1,4))

print("ROC-AUC                 :", round(roc_auc,4))

print("Scaler Saved            : intrusion_scaler.joblib")

print("Model Saved             : one_class_svm_model.joblib")

print("Label Encoders Saved    : label_encoders.joblib")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Intrusion Detection using One-Class SVM Completed Successfully!")

print("="*70)

