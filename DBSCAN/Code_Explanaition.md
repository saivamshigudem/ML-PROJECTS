# ============================================================
# Anomaly Detection using DBSCAN
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
print("Target Distribution")
print("="*70)

print(df["Class"].value_counts())

# ============================================================
# Fraud Percentage
# ============================================================

fraud_percentage = (

    df["Class"].value_counts(normalize=True)

    * 100

)

print("\n")

print("="*70)
print("Fraud Percentage")
print("="*70)

print(fraud_percentage)

# ============================================================
# Class Distribution Plot
# ============================================================

plt.figure(figsize=(6,4))

sns.countplot(

    x="Class",

    data=df

)

plt.title("Normal vs Fraud Transactions")

plt.xlabel("Class")

plt.ylabel("Count")

plt.show()

# ============================================================
# Transaction Amount Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    df["Amount"],

    bins=50,

    kde=True

)

plt.title("Transaction Amount Distribution")

plt.xlabel("Amount")

plt.show()

# ============================================================
# Transaction Time Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    df["Time"],

    bins=50,

    kde=True

)

plt.title("Transaction Time Distribution")

plt.xlabel("Time")

plt.show()

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Histograms")
print("="*70)

df.hist(

    figsize=(20,20),

    bins=30

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

plt.figure(figsize=(18,14))

sns.heatmap(

    df.corr(

        numeric_only=True

    ),

    cmap="coolwarm"

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
# Pairplot (Sampled Data)
# ============================================================

print("\n")

print("="*70)
print("Pairplot (Random Sample)")
print("="*70)

sample_df = df.sample(

    n=1000,

    random_state=42

)

sns.pairplot(

    sample_df,

    vars=["Time","Amount","V1","V2","V3"],

    hue="Class"

)

plt.show()

# ============================================================
# Correlation with Target
# ============================================================

print("\n")

print("="*70)
print("Correlation with Target")
print("="*70)

target_corr = df.corr(

    numeric_only=True

)["Class"].sort_values(

    ascending=False

)

display(target_corr)

# ============================================================
# Top 10 Highest Transactions
# ============================================================

print("\n")

print("="*70)
print("Top 10 Highest Transaction Amounts")
print("="*70)

display(

    df.nlargest(

        10,

        "Amount"

    )

)

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

print(df.isnull().sum())

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

print(df.isnull().sum())

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
# Convert Scaled Data to DataFrame
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
# Check Mean After Scaling
# ============================================================

print("\n")

print("="*70)
print("Mean of Scaled Features")
print("="*70)

display(

    X_scaled_df.mean().round(2)

)

# ============================================================
# Check Standard Deviation After Scaling
# ============================================================

print("\n")

print("="*70)
print("Standard Deviation of Scaled Features")
print("="*70)

display(

    X_scaled_df.std().round(2)

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Dataset Summary")
print("="*70)

print("Total Records          :", X.shape[0])

print("Total Features         :", X.shape[1])

print("Normal Transactions    :", y.value_counts()[0])

print("Fraud Transactions     :", y.value_counts()[1])

print("Scaled Data Shape      :", X_scaled.shape)

print("\nDataset is Ready for DBSCAN Model Building.")

print("="*70)


# ============================================================
#  Model Building & Anomaly Detection
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ============================================================
# Build DBSCAN Model
# ============================================================

dbscan = DBSCAN(

    eps=2,

    min_samples=10,

    metric="euclidean"

)

# ============================================================
# Train DBSCAN Model
# ============================================================

cluster_labels = dbscan.fit_predict(

    X_scaled

)

print("\n")

print("="*70)
print("DBSCAN Model Trained Successfully")
print("="*70)

# ============================================================
# Add Cluster Labels
# ============================================================

df["Cluster"] = cluster_labels

print("\n")

print("="*70)
print("Cluster Labels Added")
print("="*70)

display(df.head())

# ============================================================
# Cluster Distribution
# ============================================================

print("\n")

print("="*70)
print("Cluster Distribution")
print("="*70)

print(

    df["Cluster"]

    .value_counts()

    .sort_index()

)

# ============================================================
# Detect Anomalies
# ============================================================

df["Anomaly"] = np.where(

    df["Cluster"] == -1,

    1,

    0

)

print("\n")

print("="*70)
print("Anomaly Detection Summary")
print("="*70)

print(

    df["Anomaly"]

    .value_counts()

)

# ============================================================
# Total Anomalies
# ============================================================

total_anomalies = df["Anomaly"].sum()

print("\n")

print("="*70)
print("Total Detected Anomalies")
print("="*70)

print("Detected Anomalies :", total_anomalies)

print("Normal Records     :", len(df)-total_anomalies)

# ============================================================
# Number of Clusters
# ============================================================

n_clusters = len(

    set(cluster_labels)

) - (1 if -1 in cluster_labels else 0)

print("\n")

print("="*70)
print("Number of Clusters")
print("="*70)

print("Clusters Formed :", n_clusters)

# ============================================================
# Silhouette Score
# ============================================================

import numpy as np
from sklearn.metrics import silhouette_score

print("\n")
print("="*70)
print("Silhouette Score (Sampled)")
print("="*70)

if n_clusters > 1:
    # Sample up to 2000 points (or fewer if dataset is smaller)
    sample_size = min(2000, len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)

    score = silhouette_score(
        X_scaled[sample_idx],
        cluster_labels[sample_idx]
    )

    print("Silhouette Score (approx):", round(score, 4))
else:
    score = None
    print("Silhouette Score cannot be calculated.")


# ============================================================
# PCA for Visualization
# ============================================================

pca = PCA(

    n_components=2,

    random_state=42

)

X_pca = pca.fit_transform(

    X_scaled

)

print("\n")

print("="*70)
print("PCA Transformation Completed")
print("="*70)

# ============================================================
# Cluster Visualization
# ============================================================

plt.figure(figsize=(10,7))

plt.scatter(

    X_pca[:,0],

    X_pca[:,1],

    c=cluster_labels,

    cmap="viridis",

    s=10

)

plt.title("DBSCAN Clustering")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(label="Cluster")

plt.show()

# ============================================================
# Highlight Anomalies
# ============================================================

plt.figure(figsize=(10,7))

plt.scatter(

    X_pca[:,0],

    X_pca[:,1],

    color="lightgray",

    s=10,

    label="Normal"

)

plt.scatter(

    X_pca[df["Anomaly"]==1,0],

    X_pca[df["Anomaly"]==1,1],

    color="red",

    s=25,

    label="Anomaly"

)

plt.title("Detected Anomalies")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend()

plt.show()

# ============================================================
# Compare with Actual Fraud Labels
# ============================================================

print("\n")

print("="*70)
print("Comparison with Actual Labels")
print("="*70)

cm = confusion_matrix(

    y,

    df["Anomaly"]

)

print("Confusion Matrix")

print(cm)

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

        df["Anomaly"],

        zero_division=0

    )

)

# ============================================================
# Actual vs Detected
# ============================================================

comparison = pd.DataFrame({

    "Actual":y,

    "Detected":df["Anomaly"]

})

print("\n")

print("="*70)
print("Actual vs Detected")
print("="*70)

display(

    comparison.head(20)

)

# ============================================================
# Sample Detected Anomalies
# ============================================================

print("\n")

print("="*70)
print("Sample Detected Anomalies")
print("="*70)

display(

    df[df["Anomaly"]==1].head(10)

)

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("DBSCAN Anomaly Detection Summary")
print("="*70)

print("Total Transactions     :", len(df))

print("Clusters Formed        :", n_clusters)

print("Detected Anomalies     :", total_anomalies)

print("Actual Fraud Cases     :", y.sum())

print("Normal Transactions    :", len(df)-total_anomalies)

if score is not None:

    print("Silhouette Score       :", round(score,4))

else:

    print("Silhouette Score       : Not Applicable")

print("Model Status           : Ready for Deployment")

print("="*70)

# ============================================================
#  Deployment & Model Evaluation
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# Train KNN Model on DBSCAN Labels
# ============================================================

knn_model = KNeighborsClassifier(

    n_neighbors=5

)

knn_model.fit(

    X_scaled,

    df["Anomaly"]

)

print("\n")

print("="*70)

print("KNN Model Trained Successfully")

print("="*70)

# ============================================================
# Predict New Transaction
# ============================================================

new_transaction = X.iloc[[0]].copy()

new_transaction_scaled = scaler.transform(

    new_transaction

)

prediction = knn_model.predict(

    new_transaction_scaled

)

print("\n")

print("="*70)

print("Prediction for New Transaction")

print("="*70)

if prediction[0] == 1:

    print("Prediction : Anomaly Transaction")

else:

    print("Prediction : Normal Transaction")

# ============================================================
# Prediction Probability
# ============================================================

probability = knn_model.predict_proba(

    new_transaction_scaled

)

print("\n")

print("="*70)

print("Prediction Probability")

print("="*70)

print(probability)

# ============================================================
# Save Models
# ============================================================

joblib.dump(

    scaler,

    "dbscan_scaler.joblib"

)

joblib.dump(

    pca,

    "dbscan_pca.joblib"

)

joblib.dump(

    knn_model,

    "dbscan_knn_predictor.joblib"

)

print("\n")

print("="*70)

print("Models Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Models
# ============================================================

loaded_scaler = joblib.load(

    "dbscan_scaler.joblib"

)

loaded_pca = joblib.load(

    "dbscan_pca.joblib"

)

loaded_knn = joblib.load(

    "dbscan_knn_predictor.joblib"

)

print("\n")

print("="*70)

print("Saved Models Loaded Successfully")

print("="*70)

# ============================================================
# Predict Using Loaded Models
# ============================================================

loaded_prediction = loaded_knn.predict(

    loaded_scaler.transform(

        new_transaction

    )

)

print("\n")

print("="*70)

print("Prediction Using Loaded Models")

print("="*70)

if loaded_prediction[0] == 1:

    print("Prediction : Anomaly Transaction")

else:

    print("Prediction : Normal Transaction")

# ============================================================
# Sample Anomaly Transactions
# ============================================================

print("\n")

print("="*70)

print("Sample Detected Anomalies")

print("="*70)

display(

    df[df["Anomaly"] == 1].head(10)

)

# ============================================================
# Sample Normal Transactions
# ============================================================

print("\n")

print("="*70)

print("Sample Normal Transactions")

print("="*70)

display(

    df[df["Anomaly"] == 0].head(10)

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("Total Transactions       :", len(df))

print("Detected Anomalies       :", df["Anomaly"].sum())

print("Normal Transactions      :", len(df)-df["Anomaly"].sum())

print("Actual Fraud Transactions:", y.sum())

print("\nPotential Uses")

print("- Credit Card Fraud Detection")

print("- Banking Risk Monitoring")

print("- Financial Crime Detection")

print("- Payment Security")

print("- Suspicious Transaction Detection")

# ============================================================
# Export Results
# ============================================================

df.to_csv(

    "DBSCAN_Anomaly_Detection_Output.csv",

    index=False

)

print("\n")

print("="*70)

print("Output CSV Exported Successfully")

print("="*70)

# ============================================================
# Final Visualization
# ============================================================

plt.figure(figsize=(10,7))

plt.scatter(

    X_pca[df["Anomaly"]==0,0],

    X_pca[df["Anomaly"]==0,1],

    c="blue",

    s=10,

    label="Normal"

)

plt.scatter(

    X_pca[df["Anomaly"]==1,0],

    X_pca[df["Anomaly"]==1,1],

    c="red",

    s=20,

    label="Anomaly"

)

plt.title("Final DBSCAN Anomaly Detection")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend()

plt.show()

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Anomaly Detection using DBSCAN")

print("="*70)

print("Algorithm Used            : DBSCAN")

print("Deployment Model          : K-Nearest Neighbors")

print("Total Transactions        :", len(df))

print("Clusters Formed           :", n_clusters)

print("Detected Anomalies        :", df["Anomaly"].sum())

print("Actual Fraud Transactions :", y.sum())

print("Scaler File               : dbscan_scaler.joblib")

print("PCA File                  : dbscan_pca.joblib")

print("Prediction Model          : dbscan_knn_predictor.joblib")

print("Output Dataset            : DBSCAN_Anomaly_Detection_Output.csv")

print("Project Status            : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Anomaly Detection using DBSCAN Completed Successfully!")

print("="*70)

