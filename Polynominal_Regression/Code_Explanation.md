# ============================================================
# Demand Forecasting using Polynomial Regression
# Part 1 - Data Loading & Exploratory Data Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.set_style("whitegrid")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("demand_forecasting_dataset.csv")

# ============================================================
# Basic Exploration
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
# Convert Date Column
# ============================================================

df["date"] = pd.to_datetime(df["date"])

print("\nDate Converted Successfully")

# ============================================================
# Extract Date Features
# ============================================================

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

print(df.head())

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
# Distribution of Numerical Features
# ============================================================

numerical_columns = [

    "historical_sales",

    "price",

    "economic_index",

    "target_demand"

]

for col in numerical_columns:

    plt.figure(figsize=(7,4))

    sns.histplot(

        df[col],

        kde=True

    )

    plt.title(f"Distribution of {col}")

    plt.show()

# ============================================================
# Boxplots
# ============================================================

for col in numerical_columns:

    plt.figure(figsize=(7,2))

    sns.boxplot(

        x=df[col]

    )

    plt.title(col)

    plt.show()

# ============================================================
# Relationship with Target
# ============================================================

features = [

    "historical_sales",

    "price",

    "economic_index"

]

for feature in features:

    plt.figure(figsize=(7,5))

    sns.scatterplot(

        data=df,

        x=feature,

        y="target_demand"

    )

    plt.title(f"{feature} vs Target Demand")

    plt.show()

# ============================================================
# Pair Plot
# ============================================================

sns.pairplot(

    df[

        [

            "historical_sales",

            "price",

            "economic_index",

            "target_demand"

        ]

    ]

)

plt.show()

# ============================================================
# Feature Selection
# ============================================================

X = df[

    [

        "historical_sales",

        "price",

        "promotion_flag",

        "holiday_flag",

        "economic_index"

    ]

]

y = df["target_demand"]

print("\n")

print("="*60)
print("Selected Features")
print("="*60)

print(X.head())

print("\n")

print("="*60)
print("Target Variable")
print("="*60)

print(y.head())

print("\n")

print("Input Shape :", X.shape)

print("Target Shape:", y.shape)

print("\n")

print("="*60)


# ============================================================
# Split Dataset
# ============================================================

from sklearn.model_selection import train_test_split

# Split data into 80% training and 20% testing

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("="*60)
print("Training Shape")
print("="*60)

print(X_train.shape)
print(y_train.shape)

print("\n")

print("="*60)
print("Testing Shape")
print("="*60)

print(X_test.shape)
print(y_test.shape)

# ============================================================
# Baseline Model - Linear Regression
# ============================================================

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(

    X_train,

    y_train

)

# Predictions

linear_train_predictions = linear_model.predict(X_train)

linear_test_predictions = linear_model.predict(X_test)

# ============================================================
# Polynomial Feature Generation
# ============================================================

from sklearn.preprocessing import PolynomialFeatures

# Degree 2 polynomial features

poly = PolynomialFeatures(

    degree=2,

    include_bias=False

)

# Learn feature combinations from training data

X_train_poly = poly.fit_transform(X_train)

# Transform test data using the same mapping

X_test_poly = poly.transform(X_test)

print("\n")

print("="*60)
print("Original Features")
print("="*60)

print(X_train.shape)

print("\n")

print("="*60)
print("Polynomial Features")
print("="*60)

print(X_train_poly.shape)

# ============================================================
# Polynomial Regression Model
# ============================================================

poly_model = LinearRegression()

poly_model.fit(

    X_train_poly,

    y_train

)

print("\nPolynomial Regression Model Trained Successfully!")

# ============================================================
# Predictions
# ============================================================

poly_train_predictions = poly_model.predict(

    X_train_poly

)

poly_test_predictions = poly_model.predict(

    X_test_poly

)

# ============================================================
# Evaluation Metrics
# ============================================================

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)

# ------------------------------------------------------------
# Linear Regression Performance
# ------------------------------------------------------------

linear_train_r2 = r2_score(

    y_train,

    linear_train_predictions

)

linear_test_r2 = r2_score(

    y_test,

    linear_test_predictions

)

# ------------------------------------------------------------
# Polynomial Regression Performance
# ------------------------------------------------------------

poly_train_r2 = r2_score(

    y_train,

    poly_train_predictions

)

poly_test_r2 = r2_score(

    y_test,

    poly_test_predictions

)

poly_mae = mean_absolute_error(

    y_test,

    poly_test_predictions

)

poly_mse = mean_squared_error(

    y_test,

    poly_test_predictions

)

poly_rmse = np.sqrt(poly_mse)

# ============================================================
# Display Results
# ============================================================

print("\n")

print("="*70)
print("Linear Regression")
print("="*70)

print(f"Train R2 : {linear_train_r2:.4f}")

print(f"Test R2  : {linear_test_r2:.4f}")

print("\n")

print("="*70)
print("Polynomial Regression")
print("="*70)

print(f"Train R2 : {poly_train_r2:.4f}")

print(f"Test R2  : {poly_test_r2:.4f}")

print(f"MAE      : {poly_mae:.2f}")

print(f"MSE      : {poly_mse:.2f}")

print(f"RMSE     : {poly_rmse:.2f}")

# ============================================================
# Cross Validation
# ============================================================

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(

    poly_model,

    poly.fit_transform(X),

    y,

    cv=5,

    scoring="r2"

)

print("\n")

print("="*70)
print("Cross Validation")
print("="*70)

print(cv_scores)

print("\nAverage Score")

print(cv_scores.mean())

# ============================================================
# Compare Models
# ============================================================

comparison = pd.DataFrame({

    "Model":[

        "Linear Regression",

        "Polynomial Regression"

    ],

    "Train R2":[

        linear_train_r2,

        poly_train_r2

    ],

    "Test R2":[

        linear_test_r2,

        poly_test_r2

    ]

})

print("\n")

print("="*70)
print("Model Comparison")
print("="*70)

print(comparison)

# ============================================================
# Best Model
# ============================================================

if poly_test_r2 > linear_test_r2:

    print("\nPolynomial Regression performed better!")

else:

    print("\nLinear Regression performed better!")

print("\n")

print("="*70)
print("Part 2 Completed Successfully")
print("="*70)


# ============================================================
# Actual vs Predicted Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,

    poly_test_predictions,

    alpha=0.7

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    color="red",

    linewidth=2

)

plt.xlabel("Actual Demand")

plt.ylabel("Predicted Demand")

plt.title("Actual vs Predicted Demand")

plt.show()


# ============================================================
# Residual Analysis
# ============================================================

# Residual = Actual - Predicted

residuals = y_test - poly_test_predictions

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

    poly_test_predictions,

    residuals,

    alpha=0.6

)

plt.axhline(

    y=0,

    color="red",

    linestyle="--"

)

plt.xlabel("Predicted Demand")

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

plt.title("QQ Plot of Residuals")

plt.show()


# ============================================================
# Residual Skewness
# ============================================================

print("\nResidual Skewness")

print(residuals.skew())


# ============================================================
# Compare Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual Demand": y_test.values,

    "Predicted Demand": poly_test_predictions

})

print("\n")

print("="*60)
print("Sample Predictions")
print("="*60)

print(comparison.head(10))


# ============================================================
# Predict Demand for New Data
# ============================================================

# Example input
# Make sure the column order matches X_train

new_data = pd.DataFrame({

    "historical_sales":[25],

    "price":[20.5],

    "promotion_flag":[1],

    "holiday_flag":[0],

    "economic_index":[95.5]

})

# Convert to polynomial features

new_data_poly = poly.transform(new_data)

# Predict demand

predicted_demand = poly_model.predict(new_data_poly)

print("\n")

print("="*60)
print("Predicted Demand")
print("="*60)

print(predicted_demand)


# ============================================================
# Save the Trained Model
# ============================================================

from joblib import dump

# Save Polynomial Regression model

dump(

    poly_model,

    "polynomial_regression_model.joblib"

)

# Save Polynomial Feature Transformer

dump(

    poly,

    "polynomial_features.joblib"

)

print("\nModel Saved Successfully!")


# ============================================================
# Load Saved Model
# ============================================================

from joblib import load

loaded_model = load(

    "polynomial_regression_model.joblib"

)

loaded_poly = load(

    "polynomial_features.joblib"

)

print("Saved Model Loaded Successfully!")


# ============================================================
# Prediction Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    loaded_poly.transform(new_data)

)

print("\nPrediction Using Loaded Model")

print(loaded_prediction)


# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Demand Forecasting using Polynomial Regression")
print("="*70)

print(f"Training Samples          : {len(X_train)}")

print(f"Testing Samples           : {len(X_test)}")

print(f"Original Features         : {X_train.shape[1]}")

print(f"Polynomial Features       : {X_train_poly.shape[1]}")

print(f"\nTrain R2 Score           : {poly_train_r2:.4f}")

print(f"Test R2 Score            : {poly_test_r2:.4f}")

print(f"Cross Validation Score   : {cv_scores.mean():.4f}")

print(f"MAE                      : {poly_mae:.2f}")

print(f"RMSE                     : {poly_rmse:.2f}")

print("\nPolynomial Degree        : 2")

print("\nModel Ready for Deployment!")

print("="*70)

print("\nProject Completed Successfully!")
