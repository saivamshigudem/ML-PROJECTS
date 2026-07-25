# ============================================================
# Data Visualization using t-SNE & UMAP
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

df = pd.read_csv("mnist_train.csv")

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

display(df.columns.tolist())

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

print("Total Numerical Columns :", len(numerical_columns))

# ============================================================
# Target Distribution
# ============================================================

print("\n")

print("="*70)
print("Digit Distribution")
print("="*70)

display(df["label"].value_counts().sort_index())

plt.figure(figsize=(8,5))

sns.countplot(

    x="label",

    data=df

)

plt.title("Digit Distribution")

plt.xlabel("Digit")

plt.ylabel("Count")

plt.show()

# ============================================================
# Display Sample Digit Images
# ============================================================

print("\n")

print("="*70)
print("Sample Digit Images")
print("="*70)

plt.figure(figsize=(12,6))

for i in range(10):

    plt.subplot(2,5,i+1)

    image = df.iloc[i,1:].values.reshape(28,28)

    plt.imshow(image, cmap="gray")

    plt.title(f"Label : {df.iloc[i,0]}")

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Pixel Value Distribution
# ============================================================

print("\n")

print("="*70)
print("Pixel Value Distribution")
print("="*70)

plt.figure(figsize=(8,5))

sns.histplot(

    df.iloc[:,1:].values.flatten(),

    bins=50,

    kde=True

)

plt.title("Distribution of Pixel Intensities")

plt.xlabel("Pixel Value")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# Correlation Matrix (Sample Features)
# ============================================================

print("\n")

print("="*70)
print("Correlation Matrix (First 30 Pixels)")
print("="*70)

sample_features = df.iloc[:,1:31]

plt.figure(figsize=(12,10))

sns.heatmap(

    sample_features.corr(),

    cmap="coolwarm"

)

plt.title("Correlation Matrix of First 30 Pixel Features")

plt.show()

# ============================================================
# Average Image
# ============================================================

print("\n")

print("="*70)
print("Average Digit Image")
print("="*70)

average_image = df.iloc[:,1:].mean().values.reshape(28,28)

plt.figure(figsize=(5,5))

plt.imshow(

    average_image,

    cmap="gray"

)

plt.title("Average Image")

plt.axis("off")

plt.show()

# ============================================================
# Average Pixel Intensity Per Digit
# ============================================================

print("\n")

print("="*70)
print("Average Pixel Intensity Per Digit")
print("="*70)

digit_mean = df.groupby("label").mean().mean(axis=1)

display(digit_mean)

plt.figure(figsize=(8,5))

sns.barplot(

    x=digit_mean.index,

    y=digit_mean.values

)

plt.title("Average Pixel Intensity by Digit")

plt.xlabel("Digit")

plt.ylabel("Average Intensity")

plt.show()

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Dataset Summary")
print("="*70)

print("Total Records       :", df.shape[0])

print("Total Features      :", df.shape[1]-1)

print("Target Classes      :", df["label"].nunique())

print("Image Size          : 28 x 28")

print("Pixels Per Image    : 784")

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

print("Scaled Dataset Shape :", X_scaled.shape)

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
# Create Sample Dataset
# ============================================================

sample_size = 5000

X_sample = X_scaled_df.sample(

    n=sample_size,

    random_state=42

)

y_sample = y.loc[

    X_sample.index

]

print("\n")

print("="*70)
print("Sample Dataset Created")
print("="*70)

print("Sample Feature Shape :", X_sample.shape)

print("Sample Target Shape  :", y_sample.shape)

# ============================================================
# Reset Index
# ============================================================

X_sample.reset_index(

    drop=True,

    inplace=True

)

y_sample.reset_index(

    drop=True,

    inplace=True

)

# ============================================================
# Sample Class Distribution
# ============================================================

print("\n")

print("="*70)
print("Sample Class Distribution")
print("="*70)

display(

    y_sample.value_counts().sort_index()

)

plt.figure(figsize=(8,5))

sns.countplot(

    x=y_sample

)

plt.title("Digit Distribution in Sample Dataset")

plt.xlabel("Digit")

plt.ylabel("Count")

plt.show()

# ============================================================
# Visualize Random Sample Images
# ============================================================

print("\n")

print("="*70)
print("Random Sample Images")
print("="*70)

plt.figure(figsize=(12,6))

random_indices = np.random.choice(

    X_sample.index,

    10,

    replace=False

)

for i, idx in enumerate(random_indices):

    plt.subplot(2,5,i+1)

    image = X_sample.iloc[idx].values.reshape(28,28)

    plt.imshow(

        image,

        cmap="gray"

    )

    plt.title(

        f"Label : {y_sample.iloc[idx]}"

    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# Feature Statistics
# ============================================================

feature_stats = pd.DataFrame({

    "Mean": X_sample.mean(),

    "Std": X_sample.std(),

    "Min": X_sample.min(),

    "Max": X_sample.max()

})

print("\n")

print("="*70)
print("Feature Statistics")
print("="*70)

display(

    feature_stats.head(10)

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Dataset Summary")
print("="*70)

print("Original Records       :", X.shape[0])

print("Original Features      :", X.shape[1])

print("Scaled Dataset Shape   :", X_scaled.shape)

print("Sample Records         :", X_sample.shape[0])

print("Sample Features        :", X_sample.shape[1])

print("Target Classes         :", y_sample.nunique())

print("\nDataset is Ready for PCA, t-SNE and UMAP.")

print("="*70)

# ============================================================
# PCA, t-SNE & UMAP Visualization
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import time

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import umap.umap_ as umap

# ============================================================
# PCA Visualization (Baseline)
# ============================================================

start_time = time.time()

pca = PCA(

    n_components=2,

    random_state=42

)

X_pca = pca.fit_transform(

    X_sample

)

pca_time = time.time() - start_time

print("\n")

print("="*70)

print("PCA Completed Successfully")

print("="*70)

print("Execution Time :", round(pca_time,2),"Seconds")

# ============================================================
# PCA DataFrame
# ============================================================

pca_df = pd.DataFrame(

    X_pca,

    columns=["PC1","PC2"]

)

pca_df["Label"] = y_sample

display(

    pca_df.head()

)

# ============================================================
# PCA Explained Variance
# ============================================================

print("\n")

print("="*70)

print("PCA Explained Variance")

print("="*70)

for i,value in enumerate(

    pca.explained_variance_ratio_

):

    print(

        f"PC{i+1} : {value:.4f}"

    )

print(

    "\nTotal Variance :",

    round(

        pca.explained_variance_ratio_.sum(),

        4

    )

)

# ============================================================
# PCA Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=pca_df,

    x="PC1",

    y="PC2",

    hue="Label",

    palette="tab10",

    s=25,

    alpha=0.8

)

plt.title("PCA Visualization")

plt.show()

# ============================================================
# t-SNE
# ============================================================

start_time = time.time()

tsne = TSNE(

    n_components=2,

    perplexity=30,

    learning_rate="auto",

    init="pca",

    random_state=42

)

X_tsne = tsne.fit_transform(

    X_sample

)

tsne_time = time.time() - start_time

print("\n")

print("="*70)

print("t-SNE Completed Successfully")

print("="*70)

print("Execution Time :", round(tsne_time,2),"Seconds")

# ============================================================
# t-SNE DataFrame
# ============================================================

tsne_df = pd.DataFrame(

    X_tsne,

    columns=["Dim1","Dim2"]

)

tsne_df["Label"] = y_sample

display(

    tsne_df.head()

)

# ============================================================
# t-SNE Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=tsne_df,

    x="Dim1",

    y="Dim2",

    hue="Label",

    palette="tab10",

    s=25,

    alpha=0.8

)

plt.title("t-SNE Visualization")

plt.show()

# ============================================================
# UMAP
# ============================================================

start_time = time.time()

umap_model = umap.UMAP(

    n_components=2,

    n_neighbors=15,

    min_dist=0.1,

    random_state=42

)

X_umap = umap_model.fit_transform(

    X_sample

)

umap_time = time.time() - start_time

print("\n")

print("="*70)

print("UMAP Completed Successfully")

print("="*70)

print("Execution Time :", round(umap_time,2),"Seconds")

# ============================================================
# UMAP DataFrame
# ============================================================

umap_df = pd.DataFrame(

    X_umap,

    columns=["UMAP1","UMAP2"]

)

umap_df["Label"] = y_sample

display(

    umap_df.head()

)

# ============================================================
# UMAP Visualization
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=umap_df,

    x="UMAP1",

    y="UMAP2",

    hue="Label",

    palette="tab10",

    s=25,

    alpha=0.8

)

plt.title("UMAP Visualization")

plt.show()

# ============================================================
# Execution Time Comparison
# ============================================================

comparison = pd.DataFrame({

    "Algorithm":[

        "PCA",

        "t-SNE",

        "UMAP"

    ],

    "Execution Time (Seconds)":[

        round(pca_time,2),

        round(tsne_time,2),

        round(umap_time,2)

    ]

})

print("\n")

print("="*70)

print("Execution Time Comparison")

print("="*70)

display(comparison)

# ============================================================
# Execution Time Bar Chart
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(

    data=comparison,

    x="Algorithm",

    y="Execution Time (Seconds)"

)

plt.title("Execution Time Comparison")

plt.show()

# ============================================================
# Dataset Comparison
# ============================================================

comparison_shape = pd.DataFrame({

    "Description":[

        "Original Features",

        "Reduced Features"

    ],

    "Value":[

        X_sample.shape[1],

        X_pca.shape[1]

    ]

})

print("\n")

print("="*70)

print("Dimension Reduction Summary")

print("="*70)

display(comparison_shape)

# ============================================================
# Algorithm Summary
# ============================================================

summary = pd.DataFrame({

    "Algorithm":[

        "PCA",

        "t-SNE",

        "UMAP"

    ],

    "Type":[

        "Linear",

        "Non-Linear",

        "Non-Linear"

    ],

    "Supports New Data":[

        "Yes",

        "No",

        "Yes"

    ],

    "Visualization":[

        "Good",

        "Excellent",

        "Excellent"

    ]

})

print("\n")

print("="*70)

print("Algorithm Comparison")

print("="*70)

display(summary)

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)

print("Visualization Summary")

print("="*70)

print("Original Dimensions :", X_sample.shape[1])

print("Reduced Dimensions  :", X_pca.shape[1])

print("PCA Time            :", round(pca_time,2),"Seconds")

print("t-SNE Time          :", round(tsne_time,2),"Seconds")

print("UMAP Time           :", round(umap_time,2),"Seconds")

print("="*70)

# ============================================================
#  Deployment & Evaluation
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

    "visualization_scaler.joblib"

)

joblib.dump(

    pca,

    "pca_visualization_model.joblib"

)

joblib.dump(

    umap_model,

    "umap_visualization_model.joblib"

)

print("\n")

print("="*70)

print("Models Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Models
# ============================================================

loaded_scaler = joblib.load(

    "visualization_scaler.joblib"

)

loaded_pca = joblib.load(

    "pca_visualization_model.joblib"

)

loaded_umap = joblib.load(

    "umap_visualization_model.joblib"

)

print("\n")

print("="*70)

print("Saved Models Loaded Successfully")

print("="*70)

# ============================================================
# Transform New Sample
# ============================================================

new_sample = X.iloc[[0]]

new_sample_scaled = loaded_scaler.transform(

    new_sample

)

new_sample_pca = loaded_pca.transform(

    new_sample_scaled

)

new_sample_umap = loaded_umap.transform(

    new_sample_scaled

)

print("\n")

print("="*70)

print("PCA Transformation")

print("="*70)

display(

    pd.DataFrame(

        new_sample_pca,

        columns=["PC1","PC2"]

    )

)

print("\n")

print("="*70)

print("UMAP Transformation")

print("="*70)

display(

    pd.DataFrame(

        new_sample_umap,

        columns=["UMAP1","UMAP2"]

    )

)

# ============================================================
# t-SNE Information
# ============================================================

print("\n")

print("="*70)

print("t-SNE Deployment Information")

print("="*70)

print("t-SNE cannot transform unseen samples.")

print("A new t-SNE model must be fitted again.")

# ============================================================
# Export PCA Embedding
# ============================================================

pca_output = pd.DataFrame(

    X_pca,

    columns=["PC1","PC2"]

)

pca_output["Label"] = y_sample.values

pca_output.to_csv(

    "PCA_Visualization_Output.csv",

    index=False

)

# ============================================================
# Export t-SNE Embedding
# ============================================================

tsne_output = pd.DataFrame(

    X_tsne,

    columns=["Dim1","Dim2"]

)

tsne_output["Label"] = y_sample.values

tsne_output.to_csv(

    "TSNE_Visualization_Output.csv",

    index=False

)

# ============================================================
# Export UMAP Embedding
# ============================================================

umap_output = pd.DataFrame(

    X_umap,

    columns=["UMAP1","UMAP2"]

)

umap_output["Label"] = y_sample.values

umap_output.to_csv(

    "UMAP_Visualization_Output.csv",

    index=False

)

print("\n")

print("="*70)

print("Visualization Files Exported Successfully")

print("="*70)

# ============================================================
# Original vs Reduced Dimensions
# ============================================================

comparison = pd.DataFrame({

    "Description":[

        "Original Dimensions",

        "Reduced Dimensions",

        "Dimensions Removed"

    ],

    "Value":[

        X_sample.shape[1],

        X_pca.shape[1],

        X_sample.shape[1]-X_pca.shape[1]

    ]

})

print("\n")

print("="*70)

print("Dimension Reduction Summary")

print("="*70)

display(comparison)

# ============================================================
# Visualization Summary
# ============================================================

visualization_summary = pd.DataFrame({

    "Algorithm":[

        "PCA",

        "t-SNE",

        "UMAP"

    ],

    "Type":[

        "Linear",

        "Non-Linear",

        "Non-Linear"

    ],

    "Can Transform New Data":[

        "Yes",

        "No",

        "Yes"

    ],

    "Execution Time (Seconds)":[

        round(pca_time,2),

        round(tsne_time,2),

        round(umap_time,2)

    ]

})

print("\n")

print("="*70)

print("Algorithm Comparison")

print("="*70)

display(visualization_summary)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- PCA provides a fast linear projection for dimensionality reduction.")

print("- t-SNE creates highly separated clusters for visualization but cannot be reused for new samples.")

print("- UMAP produces meaningful embeddings while supporting transformation of unseen data.")

print("- UMAP is suitable for production systems because its learned embedding can be reused.")

print("- Visualization techniques help understand hidden structures in high-dimensional datasets.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Data Visualization using t-SNE & UMAP")

print("="*70)

print("Dataset                  : MNIST")

print("Original Features        :", X_sample.shape[1])

print("Reduced Features         :", X_pca.shape[1])

print("Scaler File              : visualization_scaler.joblib")

print("PCA Model                : pca_visualization_model.joblib")

print("UMAP Model               : umap_visualization_model.joblib")

print("PCA Output               : PCA_Visualization_Output.csv")

print("t-SNE Output             : TSNE_Visualization_Output.csv")

print("UMAP Output              : UMAP_Visualization_Output.csv")

print("Project Status           : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Data Visualization using t-SNE & UMAP Completed Successfully!")

print("="*70)



