# ============================================================
# Market Segmentation using Hierarchical Clustering
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

df = pd.read_csv("Wholesale customers data.csv")

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
# Channel Distribution
# ============================================================

print("\n")

print("="*70)
print("Channel Distribution")
print("="*70)

print(df["Channel"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    x="Channel",

    data=df

)

plt.title("Channel Distribution")

plt.show()

# ============================================================
# Region Distribution
# ============================================================

print("\n")

print("="*70)
print("Region Distribution")
print("="*70)

print(df["Region"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    x="Region",

    data=df

)

plt.title("Region Distribution")

plt.show()

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Histograms")
print("="*70)

df.hist(

    figsize=(18,12),

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

plt.figure(figsize=(10,8))

sns.heatmap(

    df.corr(

        numeric_only=True

    ),

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

    hue="Channel"

)

plt.show()

# ============================================================
# Purchase Distribution
# ============================================================

purchase_columns = [

    "Fresh",

    "Milk",

    "Grocery",

    "Frozen",

    "Detergents_Paper",

    "Delicassen"

]

for column in purchase_columns:

    plt.figure(figsize=(8,4))

    sns.histplot(

        df[column],

        bins=20,

        kde=True

    )

    plt.title(f"{column} Distribution")

    plt.show()
# ============================================================
#  Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

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
# Linkage Matrix
# ============================================================

linkage_matrix = linkage(

    X_scaled,

    method="ward"

)

print("\n")

print("="*70)
print("Linkage Matrix Created")
print("="*70)

# ============================================================
# Dendrogram
# ============================================================

plt.figure(figsize=(15,7))

dendrogram(

    linkage_matrix

)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()

# ============================================================
# Dendrogram (Truncated)
# ============================================================

plt.figure(figsize=(15,7))

dendrogram(

    linkage_matrix,

    truncate_mode="lastp",

    p=30,

    leaf_rotation=90,

    leaf_font_size=10,

    show_contracted=True

)

plt.title("Truncated Dendrogram")

plt.xlabel("Cluster")

plt.ylabel("Distance")

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
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Final Dataset Summary")
print("="*70)

print("Total Samples        :", X.shape[0])

print("Total Features       :", X.shape[1])

print("Scaled Feature Shape :", X_scaled.shape)

print("Optimal Clusters     :", optimal_clusters)

print("\nDataset is Ready for Hierarchical Clustering.")

print("="*70)

# ============================================================
#  Model Building & Cluster Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# ============================================================
# Build Hierarchical Clustering Model
# ============================================================

hierarchical = AgglomerativeClustering(

    n_clusters=optimal_clusters,

    metric="euclidean",

    linkage="ward"

)

# ============================================================
# Train Hierarchical Clustering Model
# ============================================================

cluster_labels = hierarchical.fit_predict(

    X_scaled

)

print("\n")

print("="*70)
print("Hierarchical Clustering Model Trained Successfully")
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
# Cluster Centers (Mean Values)
# ============================================================

cluster_centers = df.groupby(

    "Cluster"

).mean(

    numeric_only=True

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

print(

    "Silhouette Score :",

    round(silhouette,4)

)

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

print(

    "Davies-Bouldin Index :",

    round(db_index,4)

)

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

print(

    "Calinski-Harabasz Score :",

    round(ch_score,2)

)

# ============================================================
# Cluster Summary
# ============================================================

cluster_summary = df.groupby(

    "Cluster"

).agg(

    {

        "Fresh":["mean","min","max"],

        "Milk":["mean","min","max"],

        "Grocery":["mean","min","max"],

        "Frozen":["mean","min","max"],

        "Detergents_Paper":["mean","min","max"],

        "Delicassen":["mean","min","max"]

    }

)

print("\n")

print("="*70)
print("Cluster Summary")
print("="*70)

display(cluster_summary)

# ============================================================
# Customer Count in Each Cluster
# ============================================================

cluster_count = df.groupby(

    "Cluster"

).size().reset_index(

    name="Customers"

)

print("\n")

print("="*70)
print("Cluster-wise Customer Count")
print("="*70)

display(cluster_count)

# ============================================================
# Cluster Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=df,

    x="Grocery",

    y="Milk",

    hue="Cluster",

    palette="Set2",

    s=100

)

plt.title(

    "Market Segmentation using Hierarchical Clustering"

)

plt.xlabel("Grocery")

plt.ylabel("Milk")

plt.legend()

plt.show()

# ============================================================
# Cluster Visualization (Fresh vs Grocery)
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=df,

    x="Fresh",

    y="Grocery",

    hue="Cluster",

    palette="tab10",

    s=100

)

plt.title(

    "Fresh vs Grocery by Cluster"

)

plt.xlabel("Fresh")

plt.ylabel("Grocery")

plt.legend()

plt.show()

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("Hierarchical Clustering Summary")
print("="*70)

print(

    "Total Customers      :",

    df.shape[0]

)

print(

    "Total Features       :",

    X.shape[1]

)

print(

    "Number of Clusters   :",

    optimal_clusters

)

print(

    "Silhouette Score     :",

    round(silhouette,4)

)

print(

    "Davies-Bouldin Index :",

    round(db_index,4)

)

print(

    "Calinski-Harabasz    :",

    round(ch_score,2)

)

print("\n")

print(

    "Model Status : Ready for Market Analysis"

)

print("="*70)

# ============================================================
#  Model Evaluation & Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# Train KNN Model for New Customer Prediction
# ============================================================

knn_model = KNeighborsClassifier(

    n_neighbors=5

)

knn_model.fit(

    X_scaled,

    cluster_labels

)

print("\n")

print("="*70)
print("KNN Cluster Prediction Model Created")
print("="*70)

# ============================================================
# Predict Cluster for New Customer
# ============================================================

new_customer = pd.DataFrame({

    "Channel":[1],

    "Region":[2],

    "Fresh":[12000],

    "Milk":[5000],

    "Grocery":[7000],

    "Frozen":[1500],

    "Detergents_Paper":[3000],

    "Delicassen":[1200]

})

new_customer_scaled = scaler.transform(

    new_customer

)

predicted_cluster = knn_model.predict(

    new_customer_scaled

)

print("\n")

print("="*70)
print("Prediction for New Customer")
print("="*70)

print("Predicted Cluster :", predicted_cluster[0])

# ============================================================
# Save Models
# ============================================================

joblib.dump(

    scaler,

    "hierarchical_scaler.joblib"

)

joblib.dump(

    knn_model,

    "hierarchical_cluster_predictor.joblib"

)

print("\n")

print("="*70)
print("Models Saved Successfully")
print("="*70)

# ============================================================
# Load Saved Models
# ============================================================

loaded_scaler = joblib.load(

    "hierarchical_scaler.joblib"

)

loaded_knn = joblib.load(

    "hierarchical_cluster_predictor.joblib"

)

print("\n")

print("="*70)
print("Saved Models Loaded Successfully")
print("="*70)

# ============================================================
# Prediction Using Loaded Models
# ============================================================

loaded_prediction = loaded_knn.predict(

    loaded_scaler.transform(new_customer)

)

print("\n")

print("="*70)
print("Prediction Using Loaded Models")
print("="*70)

print("Predicted Cluster :", loaded_prediction[0])

# ============================================================
# Sample Customers from Each Cluster
# ============================================================

print("\n")

print("="*70)
print("Sample Customers from Each Cluster")
print("="*70)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nCluster {cluster}")

    display(

        df[df["Cluster"] == cluster].head()

    )

# ============================================================
# Cluster-wise Statistics
# ============================================================

cluster_statistics = df.groupby(

    "Cluster"

).agg({

    "Fresh":["mean","min","max"],

    "Milk":["mean","min","max"],

    "Grocery":["mean","min","max"],

    "Frozen":["mean","min","max"],

    "Detergents_Paper":["mean","min","max"],

    "Delicassen":["mean","min","max"]

})

print("\n")

print("="*70)
print("Cluster-wise Statistics")
print("="*70)

display(cluster_statistics)

# ============================================================
# Business Interpretation
# ============================================================

business_summary = df.groupby(

    "Cluster"

)[

    [

        "Fresh",

        "Milk",

        "Grocery",

        "Frozen",

        "Detergents_Paper",

        "Delicassen"

    ]

].mean()

print("\n")

print("="*70)
print("Business Summary of Each Cluster")
print("="*70)

display(business_summary)

# ============================================================
# Final Cluster Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=df,

    x="Grocery",

    y="Milk",

    hue="Cluster",

    palette="tab10",

    s=100

)

plt.title("Final Market Segmentation")

plt.xlabel("Grocery")

plt.ylabel("Milk")

plt.legend()

plt.show()

# ============================================================
# Export Clustered Dataset
# ============================================================

df.to_csv(

    "Market_Segmentation_Output.csv",

    index=False

)

print("\n")

print("="*70)
print("Clustered Dataset Exported Successfully")
print("="*70)

# ============================================================
# Top Customers from Each Cluster
# ============================================================

print("\n")

print("="*70)
print("Top Customers from Each Cluster")
print("="*70)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nTop Customers - Cluster {cluster}")

    display(

        df[df["Cluster"] == cluster].head(10)

    )

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Market Segmentation using Hierarchical Clustering")
print("="*70)

print("Algorithm Used              : Agglomerative Hierarchical Clustering")

print("Prediction Model            : K-Nearest Neighbors")

print("Total Customers             :", df.shape[0])

print("Total Features              :", X.shape[1])

print("Optimal Clusters            :", optimal_clusters)

print("Silhouette Score            :", round(silhouette,4))

print("Davies-Bouldin Index        :", round(db_index,4))

print("Calinski-Harabasz Score     :", round(ch_score,2))

print("Scaler File                 : hierarchical_scaler.joblib")

print("Prediction Model File       : hierarchical_cluster_predictor.joblib")

print("Output Dataset              : Market_Segmentation_Output.csv")

print("Deployment Status           : Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Market Segmentation using Hierarchical Clustering Completed Successfully!")

print("="*70)
