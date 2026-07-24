# ============================================================
# Feature Analysis using Principal Component Analysis (PCA)
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

df = pd.read_csv(

    "winequality-red.csv"

)

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
print("Wine Quality Distribution")
print("="*70)

print(df["quality"].value_counts().sort_index())

plt.figure(figsize=(8,5))

sns.countplot(

    x="quality",

    data=df

)

plt.title("Wine Quality Distribution")

plt.xlabel("Quality")

plt.ylabel("Count")

plt.show()

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Feature Histograms")
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

plt.figure(figsize=(12,10))

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

sample_df = df.sample(

    min(500, len(df)),

    random_state=42

)

sns.pairplot(

    sample_df,

    vars=[

        "alcohol",

        "volatile acidity",

        "citric acid",

        "sulphates",

        "quality"

    ],

    hue="quality"

)

plt.show()

# ============================================================
# Feature Distributions
# ============================================================

feature_columns = [

    "fixed acidity",

    "volatile acidity",

    "citric acid",

    "residual sugar",

    "chlorides",

    "free sulfur dioxide",

    "total sulfur dioxide",

    "density",

    "pH",

    "sulphates",

    "alcohol"

]

print("\n")

print("="*70)
print("Feature Distributions")
print("="*70)

for column in feature_columns:

    plt.figure(figsize=(8,4))

    sns.histplot(

        df[column],

        bins=20,

        kde=True

    )

    plt.title(f"{column} Distribution")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.show()

# ============================================================
# Correlation with Target
# ============================================================

print("\n")

print("="*70)
print("Correlation with Wine Quality")
print("="*70)

target_corr = df.corr(

    numeric_only=True

)["quality"].sort_values(

    ascending=False

)

display(target_corr)

# ============================================================
# Top 10 Highest Quality Wines
# ============================================================

print("\n")

print("="*70)
print("Top 10 Highest Quality Wines")
print("="*70)

display(

    df.nlargest(

        10,

        "quality"

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

    "quality",

    axis=1

)

y = df["quality"]

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
# Mean of Scaled Features
# ============================================================

print("\n")

print("="*70)
print("Mean of Scaled Features")
print("="*70)

display(

    X_scaled_df.mean().round(2)

)

# ============================================================
# Standard Deviation of Scaled Features
# ============================================================

print("\n")

print("="*70)
print("Standard Deviation of Scaled Features")
print("="*70)

display(

    X_scaled_df.std().round(2)

)

# ============================================================
# Feature Correlation After Scaling
# ============================================================

print("\n")

print("="*70)
print("Correlation Matrix of Scaled Features")
print("="*70)

plt.figure(figsize=(12,10))

sns.heatmap(

    X_scaled_df.corr(),

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

)

plt.title("Correlation Matrix (Scaled Features)")

plt.show()

# ============================================================
# Feature Variance
# ============================================================

feature_variance = pd.DataFrame({

    "Feature": X.columns,

    "Variance": X.var().values

})

print("\n")

print("="*70)
print("Original Feature Variance")
print("="*70)

display(

    feature_variance.sort_values(

        by="Variance",

        ascending=False

    )

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Dataset Summary")
print("="*70)

print("Total Records       :", X.shape[0])

print("Total Features      :", X.shape[1])

print("Target Classes      :", y.nunique())

print("Scaled Data Shape   :", X_scaled.shape)

print("\nDataset is Ready for PCA Feature Analysis.")

print("="*70)

# ============================================================
#  PCA Feature Analysis & Visualization
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.decomposition import PCA

# ============================================================
# Apply PCA (2 Components)
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

print("PCA Applied Successfully")

print("="*70)

print("Reduced Dataset Shape :", X_pca.shape)

# ============================================================
# PCA DataFrame
# ============================================================

pca_df = pd.DataFrame(

    X_pca,

    columns=["PC1","PC2"]

)

pca_df["quality"] = y.values

print("\n")

print("="*70)

print("PCA DataFrame")

print("="*70)

display(

    pca_df.head()

)

# ============================================================
# Explained Variance Ratio
# ============================================================

explained_variance = pca.explained_variance_ratio_

print("\n")

print("="*70)

print("Explained Variance Ratio")

print("="*70)

for i, value in enumerate(explained_variance):

    print(

        f"PC{i+1} : {value:.4f}"

    )

# ============================================================
# Total Explained Variance
# ============================================================

print("\n")

print("="*70)

print("Total Explained Variance")

print("="*70)

print(

    "Total :", round(

        explained_variance.sum(),

        4

    )

)

# ============================================================
# Find Components for 95% Variance
# ============================================================

pca_95 = PCA(

    n_components=0.95,

    random_state=42

)

pca_95.fit(

    X_scaled

)

print("\n")

print("="*70)

print("Components Required for 95% Variance")

print("="*70)

print(

    "Number of Components :",

    pca_95.n_components_

)

# ============================================================
# Cumulative Explained Variance
# ============================================================

cumulative_variance = np.cumsum(

    pca.explained_variance_ratio_

)

print("\n")

print("="*70)

print("Cumulative Explained Variance")

print("="*70)

print(cumulative_variance)

# ============================================================
# Scree Plot
# ============================================================

full_pca = PCA()

full_pca.fit(

    X_scaled

)

plt.figure(figsize=(10,6))

plt.plot(

    range(

        1,

        len(full_pca.explained_variance_ratio_)+1

    ),

    full_pca.explained_variance_ratio_,

    marker="o"

)

plt.title("Scree Plot")

plt.xlabel("Principal Components")

plt.ylabel("Explained Variance Ratio")

plt.grid(True)

plt.show()

# ============================================================
# Cumulative Variance Plot
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(

    np.cumsum(

        full_pca.explained_variance_ratio_

    ),

    marker="o"

)

plt.axhline(

    y=0.95,

    color="red",

    linestyle="--",

    label="95% Variance"

)

plt.title("Cumulative Explained Variance")

plt.xlabel("Number of Components")

plt.ylabel("Cumulative Variance")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# PCA Component Matrix
# ============================================================

component_matrix = pd.DataFrame(

    pca.components_,

    columns=X.columns,

    index=["PC1","PC2"]

)

print("\n")

print("="*70)

print("PCA Component Matrix")

print("="*70)

display(

    component_matrix

)

# ============================================================
# Feature Importance
# ============================================================

feature_importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":

    np.abs(

        pca.components_[0]

    )

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)

print("Feature Importance (PC1)")

print("="*70)

display(

    feature_importance

)

# ============================================================
# Principal Component Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=pca_df,

    x="PC1",

    y="PC2",

    hue="quality",

    palette="viridis",

    s=70

)

plt.title(

    "PCA - First Two Principal Components"

)

plt.show()

# ============================================================
# Feature Contribution Plot
# ============================================================

plt.figure(figsize=(12,6))

sns.barplot(

    data=feature_importance,

    x="Importance",

    y="Feature"

)

plt.title(

    "Feature Contribution to Principal Component 1"

)

plt.show()

# ============================================================
# Original vs Reduced Dataset
# ============================================================

print("\n")

print("="*70)

print("Original vs Reduced Dataset")

print("="*70)

print(

    "Original Shape :", X.shape

)

print(

    "Reduced Shape  :", X_pca.shape

)

print(

    "Features Reduced :", 

    X.shape[1] - X_pca.shape[1]

)

# ============================================================
# PCA Summary
# ============================================================

print("\n")

print("="*70)

print("PCA Summary")

print("="*70)

print(

    "Original Features :", X.shape[1]

)

print(

    "Reduced Features :", X_pca.shape[1]

)

print(

    "95% Variance Components :",

    pca_95.n_components_

)

print(

    "Variance Captured (2 PCs) :",

    round(

        explained_variance.sum()*100,

        2

    ),

    "%"

)

print(

    "Model Status : Ready for Deployment"

)

print("="*70)
# ============================================================
#  Deployment & Feature Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save Models
# ============================================================

joblib.dump(

    scaler,

    "pca_scaler.joblib"

)

joblib.dump(

    pca,

    "pca_model.joblib"

)

print("\n")

print("="*70)

print("Models Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Models
# ============================================================

loaded_scaler = joblib.load(

    "pca_scaler.joblib"

)

loaded_pca = joblib.load(

    "pca_model.joblib"

)

print("\n")

print("="*70)

print("Saved Models Loaded Successfully")

print("="*70)

# ============================================================
# Transform New Sample
# ============================================================

new_sample = X.iloc[[0]].copy()

new_sample_scaled = loaded_scaler.transform(

    new_sample

)

new_sample_pca = loaded_pca.transform(

    new_sample_scaled

)

print("\n")

print("="*70)

print("New Sample After PCA Transformation")

print("="*70)

display(

    pd.DataFrame(

        new_sample_pca,

        columns=["PC1","PC2"]

    )

)

# ============================================================
# PCA Dataset
# ============================================================

pca_dataset = pd.DataFrame(

    X_pca,

    columns=["PC1","PC2"]

)

pca_dataset["quality"] = y.values

print("\n")

print("="*70)

print("PCA Dataset")

print("="*70)

display(

    pca_dataset.head()

)

# ============================================================
# Export PCA Dataset
# ============================================================

pca_dataset.to_csv(

    "Wine_PCA_Output.csv",

    index=False

)

print("\n")

print("="*70)

print("PCA Dataset Exported Successfully")

print("="*70)

# ============================================================
# Original vs Reduced Dataset
# ============================================================

comparison = pd.DataFrame({

    "Description":[

        "Original Features",

        "Reduced Features",

        "Reduction"

    ],

    "Value":[

        X.shape[1],

        X_pca.shape[1],

        X.shape[1]-X_pca.shape[1]

    ]

})

print("\n")

print("="*70)

print("Original vs Reduced Dataset")

print("="*70)

display(comparison)

# ============================================================
# PCA Component Statistics
# ============================================================

component_statistics = pd.DataFrame({

    "Principal Component":[

        "PC1",

        "PC2"

    ],

    "Explained Variance":

    explained_variance,

    "Cumulative Variance":

    np.cumsum(

        explained_variance

    )

})

print("\n")

print("="*70)

print("Principal Component Statistics")

print("="*70)

display(component_statistics)

# ============================================================
# Top 5 Important Features
# ============================================================

top_features = feature_importance.head(5)

print("\n")

print("="*70)

print("Top 5 Most Important Features")

print("="*70)

display(top_features)

# ============================================================
# Final PCA Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=pca_dataset,

    x="PC1",

    y="PC2",

    hue="quality",

    palette="viridis",

    s=80

)

plt.title(

    "Wine Dataset in PCA Space"

)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend(title="Quality")

plt.show()

# ============================================================
# Explained Variance Chart
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    ["PC1","PC2"],

    explained_variance

)

plt.title(

    "Explained Variance by Principal Components"

)

plt.ylabel(

    "Explained Variance Ratio"

)

plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- PCA reduced the feature dimensions while preserving maximum information.")

print("- Highly correlated variables were compressed into principal components.")

print("- Reduced dimensionality improves visualization and can speed up model training.")

print("- PCA is commonly used before clustering, classification, and anomaly detection.")

print("- Feature reduction can help reduce overfitting in some machine learning models.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Feature Analysis using Principal Component Analysis (PCA)")

print("="*70)

print("Algorithm Used              : Principal Component Analysis")

print("Original Features           :", X.shape[1])

print("Reduced Features            :", X_pca.shape[1])

print("95% Variance Components     :", pca_95.n_components_)

print("Variance Captured (2 PCs)   :", round(explained_variance.sum()*100,2), "%")

print("Scaler File                 : pca_scaler.joblib")

print("PCA Model File              : pca_model.joblib")

print("Output Dataset              : Wine_PCA_Output.csv")

print("Project Status              : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Feature Analysis using PCA Completed Successfully!")

print("="*70)
