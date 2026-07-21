# ============================================================
# Customer Segmentation using K-Means Clustering
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

df = pd.read_csv("Mall_Customers.csv")

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
# Gender Distribution
# ============================================================

print("\n")

print("="*70)
print("Gender Distribution")
print("="*70)

print(df["Gender"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    x="Gender",

    data=df

)

plt.title("Gender Distribution")

plt.show()

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Histograms")
print("="*70)

df.hist(

    figsize=(12,8),

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

plt.figure(figsize=(8,6))

sns.heatmap(

    df.corr(numeric_only=True),

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
# Pairplot
# ============================================================

print("\n")

print("="*70)
print("Pairplot")
print("="*70)

sns.pairplot(

    df,

    hue="Gender"

)

plt.show()

# ============================================================
# Annual Income Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    df["Annual Income (k$)"],

    bins=20,

    kde=True

)

plt.title("Annual Income Distribution")

plt.show()

# ============================================================
# Spending Score Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    df["Spending Score (1-100)"],

    bins=20,

    kde=True

)

plt.title("Spending Score Distribution")

plt.show()

# ============================================================
# Age Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    df["Age"],

    bins=20,

    kde=True

)

plt.title("Age Distribution")

plt.show()

# ============================================================
#  Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

# ============================================================
# Remove Unnecessary Columns
# ============================================================

columns_to_drop = [

    "CustomerID"

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
# Missing Values Before Treatment
# ============================================================

print("\n")

print("="*70)
print("Missing Values Before Treatment")
print("="*70)

print(df.isnull().sum())

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
            df[column].fillna("missing", inplace=True)  # fallback if column is all NaN

print("\n")
print("="*70)
print("Missing Values After Treatment")
print("="*70)
print(df.isnull().sum())

# ============================================================
# Encode Gender
# ============================================================

label_encoder = LabelEncoder()

df["Gender"] = label_encoder.fit_transform(

    df["Gender"]

)

print("\n")

print("="*70)
print("Gender Encoding Completed")
print("="*70)

print(df.head())

# ============================================================
# Feature Matrix
# ============================================================

X = df.copy()

print("\n")

print("="*70)
print("Feature Matrix")
print("="*70)

print("Shape :", X.shape)

display(X.head())

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

print("Scaled Shape :", X_scaled.shape)

# ============================================================
# Elbow Method
# ============================================================

wcss = []

for i in range(1,11):

    kmeans = KMeans(

        n_clusters=i,

        init="k-means++",

        random_state=42,

        n_init=10

    )

    kmeans.fit(

        X_scaled

    )

    wcss.append(

        kmeans.inertia_

    )

# ============================================================
# WCSS Values
# ============================================================

print("\n")

print("="*70)
print("WCSS Values")
print("="*70)

for i,value in enumerate(wcss,start=1):

    print(f"{i} Clusters : {value:.2f}")

# ============================================================
# Elbow Curve
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    range(1,11),

    wcss,

    marker="o"

)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()

# ============================================================
# Optimal Number of Clusters
# ============================================================

optimal_clusters = 5

print("\n")

print("="*70)
print("Optimal Number of Clusters")
print("="*70)

print("Selected Clusters :", optimal_clusters)

# ============================================================
# Final Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Final Dataset Summary")
print("="*70)

print("Total Samples        :", X.shape[0])

print("Total Features       :", X.shape[1])

print("Scaled Feature Shape :", X_scaled.shape)

print("Optimal Clusters     :", optimal_clusters)

print("\nDataset is Ready for K-Means Clustering.")

print("="*70)


# ============================================================
#  Model Building & Cluster Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# ============================================================
# Build K-Means Model
# ============================================================

kmeans = KMeans(

    n_clusters=optimal_clusters,

    init="k-means++",

    max_iter=300,

    n_init=10,

    random_state=42

)

# ============================================================
# Train K-Means Model
# ============================================================

cluster_labels = kmeans.fit_predict(

    X_scaled

)

print("\n")

print("="*70)
print("K-Means Model Trained Successfully")
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

print(df["Cluster"].value_counts().sort_index())

# ============================================================
# Cluster Centers
# ============================================================

cluster_centers = pd.DataFrame(

    scaler.inverse_transform(

        kmeans.cluster_centers_

    ),

    columns=X.columns

)

print("\n")

print("="*70)
print("Cluster Centers")
print("="*70)

display(cluster_centers)

# ============================================================
# Silhouette Score
# ============================================================

silhouette = silhouette_score(

    X_scaled,

    cluster_labels

)

print("\n")

print("="*70)
print("Silhouette Score")
print("="*70)

print("Silhouette Score :", round(silhouette,4))

# ============================================================
# Davies-Bouldin Index
# ============================================================

db_index = davies_bouldin_score(

    X_scaled,

    cluster_labels

)

print("\n")

print("="*70)
print("Davies-Bouldin Index")
print("="*70)

print("Davies-Bouldin Index :", round(db_index,4))

# ============================================================
# Calinski-Harabasz Score
# ============================================================

ch_score = calinski_harabasz_score(

    X_scaled,

    cluster_labels

)

print("\n")

print("="*70)
print("Calinski-Harabasz Score")
print("="*70)

print("Calinski-Harabasz Score :", round(ch_score,2))

# ============================================================
# Cluster Summary
# ============================================================

cluster_summary = df.groupby(

    "Cluster"

).mean(numeric_only=True)

print("\n")

print("="*70)
print("Cluster Summary")
print("="*70)

display(cluster_summary)

# ============================================================
# Customer Segmentation Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=df,

    x="Annual Income (k$)",

    y="Spending Score (1-100)",

    hue="Cluster",

    palette="Set2",

    s=100

)

plt.scatter(

    cluster_centers["Annual Income (k$)"],

    cluster_centers["Spending Score (1-100)"],

    marker="X",

    s=300,

    color="black",

    label="Centroids"

)

plt.title("Customer Segmentation using K-Means")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.show()

# ============================================================
# Cluster-wise Customer Count
# ============================================================

cluster_count = df.groupby(

    "Cluster"

).size().reset_index(name="Customers")

print("\n")

print("="*70)
print("Cluster-wise Customer Count")
print("="*70)

display(cluster_count)

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("K-Means Clustering Summary")
print("="*70)

print("Total Customers      :", df.shape[0])

print("Total Features       :", X.shape[1])

print("Number of Clusters   :", optimal_clusters)

print("Silhouette Score     :", round(silhouette,4))

print("Davies-Bouldin Index :", round(db_index,4))

print("Calinski-Harabasz    :", round(ch_score,2))

print("\n")

print("Model Status : Ready for Cluster Analysis")

print("="*70)

# ============================================================
#  Model Evaluation & Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Predict Cluster for New Customer
# ============================================================

new_customer = pd.DataFrame({

    "Gender":[1],

    "Age":[30],

    "Annual Income (k$)":[65],

    "Spending Score (1-100)":[70]

})

new_customer_scaled = scaler.transform(

    new_customer

)

predicted_cluster = kmeans.predict(

    new_customer_scaled

)

print("\n")

print("="*70)
print("Prediction for New Customer")
print("="*70)

print("Predicted Cluster :", predicted_cluster[0])

# ============================================================
# Save K-Means Model
# ============================================================

joblib.dump(

    kmeans,

    "customer_segmentation_kmeans_model.joblib"

)

print("\n")

print("="*70)
print("Model Saved Successfully")
print("="*70)

# ============================================================
# Save Scaler
# ============================================================

joblib.dump(

    scaler,

    "customer_segmentation_scaler.joblib"

)

print("\n")

print("="*70)
print("Scaler Saved Successfully")
print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = joblib.load(

    "customer_segmentation_kmeans_model.joblib"

)

loaded_scaler = joblib.load(

    "customer_segmentation_scaler.joblib"

)

print("\n")

print("="*70)
print("Saved Model Loaded Successfully")
print("="*70)

# ============================================================
# Prediction Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    loaded_scaler.transform(new_customer)

)

print("\n")

print("="*70)
print("Prediction Using Loaded Model")
print("="*70)

print("Predicted Cluster :", loaded_prediction[0])

# ============================================================
# Customers from Each Cluster
# ============================================================

print("\n")

print("="*70)
print("Sample Customers from Each Cluster")
print("="*70)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nCluster {cluster}")

    display(

        df[df["Cluster"]==cluster].head()

    )

# ============================================================
# Cluster-wise Statistics
# ============================================================

cluster_statistics = df.groupby(

    "Cluster"

).agg({

    "Age":["mean","min","max"],

    "Annual Income (k$)":["mean","min","max"],

    "Spending Score (1-100)":["mean","min","max"]

})

print("\n")

print("="*70)
print("Cluster-wise Statistics")
print("="*70)

display(cluster_statistics)

# ============================================================
# Final Customer Segmentation Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=df,

    x="Annual Income (k$)",

    y="Spending Score (1-100)",

    hue="Cluster",

    palette="tab10",

    s=100

)

plt.scatter(

    cluster_centers["Annual Income (k$)"],

    cluster_centers["Spending Score (1-100)"],

    marker="X",

    color="black",

    s=300,

    label="Centroids"

)

plt.title("Final Customer Segmentation")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.show()

# ============================================================
# Export Clustered Dataset
# ============================================================

df.to_csv(

    "Customer_Segmentation_Output.csv",

    index=False

)

print("\n")

print("="*70)
print("Clustered Dataset Exported Successfully")
print("="*70)

# ============================================================
# Top 10 Customers from Each Cluster
# ============================================================

print("\n")

print("="*70)
print("Top Customers from Each Cluster")
print("="*70)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nTop Customers - Cluster {cluster}")

    display(

        df[df["Cluster"]==cluster].head(10)

    )

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Customer Segmentation using K-Means")
print("="*70)

print("Algorithm Used              : K-Means Clustering")

print("Total Customers             :", df.shape[0])

print("Total Features              :", X.shape[1])

print("Optimal Clusters            :", optimal_clusters)

print("Silhouette Score            :", round(silhouette,4))

print("Davies-Bouldin Index        :", round(db_index,4))

print("Calinski-Harabasz Score     :", round(ch_score,2))

print("Model File                  : customer_segmentation_kmeans_model.joblib")

print("Scaler File                 : customer_segmentation_scaler.joblib")

print("Output Dataset              : Customer_Segmentation_Output.csv")

print("Deployment Status           : Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Customer Segmentation using K-Means Completed Successfully!")

print("="*70)
