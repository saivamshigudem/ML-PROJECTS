
# ============================================================
# Salary Prediction using Multiple Linear Regression
# Part 1 - Data Loading & Exploratory Data Analysis (EDA)
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import numpy as np                  # Numerical computations
import pandas as pd                 # Data manipulation and analysis
import matplotlib.pyplot as plt     # Data visualization
import seaborn as sns               # Statistical visualizations

# Display plots inside Jupyter Notebook
%matplotlib inline

# Set a clean plotting style
sns.set_style("whitegrid")

# ============================================================
# Load the Dataset
# ============================================================

# Read the salary dataset
df = pd.read_csv("salary_prediction.csv")

# ============================================================
# Basic Dataset Exploration
# ============================================================

print("=" * 60)
print("First 5 Records")
print("=" * 60)

print(df.head())

print("\n")

print("=" * 60)
print("Last 5 Records")
print("=" * 60)

print(df.tail())

print("\n")

print("=" * 60)
print("Dataset Shape")
print("=" * 60)

print(df.shape)

print("\n")

print("=" * 60)
print("Column Names")
print("=" * 60)

print(df.columns)

print("\n")

print("=" * 60)
print("Dataset Information")
print("=" * 60)

df.info()

print("\n")

print("=" * 60)
print("Data Types")
print("=" * 60)

print(df.dtypes)

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())

# ============================================================
# Duplicate Records
# ============================================================

print("\n")

print("=" * 60)
print("Duplicate Records")
print("=" * 60)

print(df.duplicated().sum())

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("=" * 60)
print("Statistical Summary")
print("=" * 60)

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

print("=" * 60)
print("Categorical Columns")
print("=" * 60)

print(categorical_columns)

print("\n")

print("=" * 60)
print("Numerical Columns")
print("=" * 60)

print(numerical_columns)

# ============================================================
# Explore Categorical Features
# ============================================================

for col in categorical_columns:

    print("\n")
    print("=" * 60)
    print(col.upper())
    print("=" * 60)

    print(df[col].value_counts())

# ============================================================
# Correlation Matrix
# ============================================================

print("\n")

print("=" * 60)
print("Correlation Matrix")
print("=" * 60)

correlation = df.corr(numeric_only=True)

print(correlation)

# Plot Correlation Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# Distribution of Numerical Features
# ============================================================

for col in numerical_columns:

    plt.figure(figsize=(7,4))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(f"Distribution of {col}")

    plt.show()

# ============================================================
# Boxplots (Outlier Detection)
# ============================================================

for col in numerical_columns:

    plt.figure(figsize=(7,2))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot of {col}")

    plt.show()

# ============================================================
# Count Plots for Categorical Features
# ============================================================

for col in categorical_columns:

    plt.figure(figsize=(8,4))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.xticks(rotation=45)

    plt.title(col)

    plt.show()

# ============================================================
# Relationship Between Numerical Features & Salary
# ============================================================

numerical_features = [
    "experience_years",
    "skills_count",
    "certifications"
]

for feature in numerical_features:

    plt.figure(figsize=(7,5))

    sns.scatterplot(
        data=df,
        x=feature,
        y="salary"
    )

    plt.title(f"{feature} vs Salary")

    plt.show()

# ============================================================
# Pairplot
# ============================================================

sns.pairplot(
    df[
        [
            "experience_years",
            "skills_count",
            "certifications",
            "salary"
        ]
    ]
)

plt.show()

# ============================================================
# Feature Selection
# ============================================================

# Remove target column to create input features

X = df.drop("salary", axis=1)

# Target variable

y = df["salary"]

print("\n")

print("=" * 60)
print("Input Features")
print("=" * 60)

print(X.head())

print("\n")

print("=" * 60)
print("Target Variable")
print("=" * 60)

print(y.head())

print("\n")

print("=" * 60)
print("Input Shape")
print("=" * 60)

print(X.shape)

print("\n")

print("=" * 60)
print("Target Shape")
print("=" * 60)

print(y.shape)

print("\n")

print("=" * 60)
# ============================================================
# Check Unique Values in Categorical Columns
# ============================================================

print("="*60)
print("Unique Values in Categorical Columns")
print("="*60)

for col in categorical_columns:

    print(f"\n{col.upper()}")

    print("-"*40)

    print(df[col].unique())

    print(f"\nTotal Unique Values : {df[col].nunique()}")


# ============================================================
# One-Hot Encoding
# ============================================================

# Machine Learning algorithms cannot understand text values.
# Convert categorical columns into numerical columns using One-Hot Encoding.

categorical_features = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work"
]

df_encoded = pd.get_dummies(

    df,

    columns=categorical_features,

    drop_first=True

)

print("\n")

print("="*60)
print("Dataset After Encoding")
print("="*60)

print(df_encoded.head())


# ============================================================
# Dataset Shape After Encoding
# ============================================================

print("\nEncoded Dataset Shape")

print(df_encoded.shape)


# ============================================================
# Check Data Types
# ============================================================

print("\nData Types After Encoding")

print(df_encoded.dtypes)


# ============================================================
# Correlation Matrix
# ============================================================

correlation = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(18,12))

sns.heatmap(

    correlation,

    cmap="coolwarm"

)

plt.title("Correlation Matrix After Encoding")

plt.show()


# ============================================================
# Separate Features & Target
# ============================================================

# Features

X = df_encoded.drop("salary", axis=1)

# Target

y = df_encoded["salary"]

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
# Feature Names
# ============================================================

print("\nTotal Features Used")

print(len(X.columns))

print("\n")

print(X.columns)


# ============================================================
# Train-Test Split
# ============================================================

from sklearn.model_selection import train_test_split

# Split dataset into:
# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\n")

print("="*60)
print("Training Dataset")
print("="*60)

print(X_train.shape)

print(y_train.shape)

print("\n")

print("="*60)
print("Testing Dataset")
print("="*60)

print(X_test.shape)

print(y_test.shape)


# ============================================================
# Verify Missing Values
# ============================================================

print("\n")

print("="*60)
print("Missing Values After Encoding")
print("="*60)

print(X_train.isnull().sum().sum())

print(X_test.isnull().sum().sum())


# ============================================================
# Verify All Features Are Numeric
# ============================================================

print("\n")

print("="*60)
print("Feature Data Types")
print("="*60)

print(X_train.dtypes.unique())


# ============================================================
# Final Dataset Ready
# ============================================================

print("\n")

print("="*60)
print("Data Preprocessing Completed Successfully")
print("="*60)

print(f"Total Features : {X_train.shape[1]}")

print(f"Training Samples : {X_train.shape[0]}")

print(f"Testing Samples : {X_test.shape[0]}")

# ============================================================
# Import Linear Regression
# ============================================================

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model

model = LinearRegression()

# ============================================================
# Train the Model
# ============================================================

# Learn the relationship between input features and salary

model.fit(X_train, y_train)

print("="*60)
print("Model Training Completed Successfully")
print("="*60)

# ============================================================
# Model Parameters
# ============================================================

# Display intercept

print("\nIntercept")

print(model.intercept_)

# Display coefficients

print("\nCoefficients")

coefficients = pd.DataFrame({

    "Feature": X_train.columns,

    "Coefficient": model.coef_

})

print(coefficients)

# ============================================================
# Training Predictions
# ============================================================

train_predictions = model.predict(X_train)

# ============================================================
# Testing Predictions
# ============================================================

test_predictions = model.predict(X_test)

# ============================================================
# Compare Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual Salary": y_test.values,

    "Predicted Salary": test_predictions

})

print("\nSample Predictions")

print(comparison.head(10))

# ============================================================
# Model Evaluation
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Training Metrics
# -----------------------------

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_mse = mean_squared_error(
    y_train,
    train_predictions
)

train_rmse = np.sqrt(train_mse)

train_r2 = r2_score(
    y_train,
    train_predictions
)

# -----------------------------
# Testing Metrics
# -----------------------------

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    test_predictions
)

# ============================================================
# Display Performance
# ============================================================

print("\n")

print("="*60)
print("Training Performance")
print("="*60)

print(f"MAE  : {train_mae:.2f}")
print(f"MSE  : {train_mse:.2f}")
print(f"RMSE : {train_rmse:.2f}")
print(f"R2   : {train_r2:.4f}")

print("\n")

print("="*60)
print("Testing Performance")
print("="*60)

print(f"MAE  : {test_mae:.2f}")
print(f"MSE  : {test_mse:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"R2   : {test_r2:.4f}")

# ============================================================
# Cross Validation
# ============================================================

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(

    model,

    X,

    y,

    cv=5,

    scoring="r2"

)

print("\n")

print("="*60)
print("Cross Validation")
print("="*60)

print("Individual Scores")

print(cv_scores)

print("\nAverage Score")

print(cv_scores.mean())

# ============================================================
# Feature Importance
# ============================================================

importance = coefficients.copy()

importance["Absolute Coefficient"] = abs(

    importance["Coefficient"]

)

importance = importance.sort_values(

    by="Absolute Coefficient",

    ascending=False

)

print("\n")

print("="*60)
print("Feature Importance")
print("="*60)

print(importance)

# ============================================================
# Top 10 Most Influential Features
# ============================================================

top10 = importance.head(10)

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
print("Multiple Linear Regression Summary")
print("="*60)

print(f"Training Samples      : {len(X_train)}")
print(f"Testing Samples       : {len(X_test)}")

print(f"\nIntercept             : {model.intercept_:.2f}")

print(f"\nTrain R2 Score        : {train_r2:.4f}")

print(f"Test R2 Score         : {test_r2:.4f}")

print(f"Cross Validation R2   : {cv_scores.mean():.4f}")

print(f"Test MAE              : {test_mae:.2f}")

print(f"Test RMSE             : {test_rmse:.2f}")

print("\nModel Trained Successfully!")

# ============================================================
# Actual vs Predicted Salary
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,

    test_predictions,

    alpha=0.6

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    color="red",

    linewidth=2

)

plt.xlabel("Actual Salary")

plt.ylabel("Predicted Salary")

plt.title("Actual vs Predicted Salary")

plt.show()


# ============================================================
# Residual Analysis
# ============================================================

# Residual = Actual - Predicted

residuals = y_test - test_predictions

print("="*60)
print("First 10 Residuals")
print("="*60)

print(residuals.head(10))


# ============================================================
# Residual Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    residuals,

    kde=True

)

plt.title("Residual Distribution")

plt.xlabel("Residual")

plt.show()


# ============================================================
# Residual Scatter Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    test_predictions,

    residuals,

    alpha=0.6

)

plt.axhline(

    y=0,

    color="red",

    linestyle="--"

)

plt.xlabel("Predicted Salary")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.show()


# ============================================================
# QQ Plot
# ============================================================

import scipy.stats as stats

plt.figure(figsize=(7,7))

stats.probplot(

    residuals,

    dist="norm",

    plot=plt

)

plt.title("QQ Plot")

plt.show()


# ============================================================
# Residual Skewness
# ============================================================

print("\nResidual Skewness")

print(residuals.skew())


# ============================================================
# Statsmodels Summary
# ============================================================

import statsmodels.api as sm

# Add intercept column

X_sm = sm.add_constant(X)

# Train OLS model

ols_model = sm.OLS(

    y,

    X_sm

).fit()

print("\n")

print("="*60)
print("OLS Regression Summary")
print("="*60)

print(ols_model.summary())


# ============================================================
# Predict Salary for New Employee
# ============================================================

# IMPORTANT:
# The new employee data must have the same columns as X_train.

new_employee = pd.DataFrame(

    [X_train.iloc[0].values],

    columns=X_train.columns

)

predicted_salary = model.predict(new_employee)

print("\n")

print("="*60)
print("Predicted Salary")
print("="*60)

print(predicted_salary)


# ============================================================
# Save Model
# ============================================================

from joblib import dump

dump(

    model,

    "salary_prediction_model.joblib"

)

print("\nModel Saved Successfully!")


# ============================================================
# Load Model
# ============================================================

from joblib import load

loaded_model = load(

    "salary_prediction_model.joblib"

)

print("Model Loaded Successfully!")


# ============================================================
# Prediction Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    new_employee

)

print("\nPrediction Using Loaded Model")

print(loaded_prediction)


# ============================================================
# Final Project Summary
# ============================================================

print("\n")
print("="*70)
print("Salary Prediction using Multiple Linear Regression")
print("="*70)

print(f"Training Samples           : {len(X_train)}")

print(f"Testing Samples            : {len(X_test)}")

print(f"Total Features             : {X_train.shape[1]}")

print(f"\nIntercept                  : {model.intercept_:.2f}")

print(f"Train R2 Score             : {train_r2:.4f}")

print(f"Test R2 Score              : {test_r2:.4f}")

print(f"Cross Validation Score     : {cv_scores.mean():.4f}")

print(f"Test MAE                   : {test_mae:.2f}")

print(f"Test RMSE                  : {test_rmse:.2f}")

print("\nModel Status : Ready for Deployment")

print("="*70)

print("\nProject Completed Successfully!")
