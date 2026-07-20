# ============================================================
# Sales Prediction using CatBoost Regressor
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

df = pd.read_csv("bigmart_sales.csv")

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
# Target Variable Distribution
# ============================================================

print("\n")

print("="*70)
print("Target Variable Distribution")
print("="*70)

display(df["Item_Outlet_Sales"].describe())

# ============================================================
# Histogram of Target Variable
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    df["Item_Outlet_Sales"],

    bins=30,

    kde=True

)

plt.title("Distribution of Item Outlet Sales")

plt.xlabel("Item Outlet Sales")

plt.show()

# ============================================================
# Histograms
# ============================================================

print("\n")

print("="*70)
print("Histograms")
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

plt.figure(figsize=(12,8))

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

    df[

        [

            "Item_Weight",

            "Item_Visibility",

            "Item_MRP",

            "Item_Outlet_Sales"

        ]

    ]

)

plt.show()

# ============================================================
# Sales Statistics
# ============================================================

print("\n")

print("="*70)
print("Sales Statistics")
print("="*70)

print("Minimum Sales :", round(df["Item_Outlet_Sales"].min(),2))

print("Maximum Sales :", round(df["Item_Outlet_Sales"].max(),2))

print("Average Sales :", round(df["Item_Outlet_Sales"].mean(),2))

print("Median Sales  :", round(df["Item_Outlet_Sales"].median(),2))

# ============================================================
#  Data Cleaning & Preprocessing
# ============================================================

# ============================================================
# Remove Unnecessary Columns
# ============================================================

columns_to_drop = [

    "Item_Identifier"

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
# Fill Numerical Missing Values
# ============================================================

numerical_columns = df.select_dtypes(

    include=["int64","float64"]

).columns

for column in numerical_columns:

    if column != "Item_Outlet_Sales":

        df[column].fillna(

            df[column].median(),

            inplace=True

        )

# ============================================================
# Fill Categorical Missing Values
# ============================================================

categorical_columns = df.select_dtypes(

    include=["object"]

).columns

for column in categorical_columns:

    df[column].fillna(

        df[column].mode()[0],

        inplace=True

    )

print("\n")

print("="*70)
print("Missing Values After Treatment")
print("="*70)

print(df.isnull().sum())

# ============================================================
# Feature Engineering
# ============================================================

current_year = 2026

df["Outlet_Age"] = current_year - df["Outlet_Establishment_Year"]

print("\n")

print("="*70)
print("Feature Engineering Completed")
print("="*70)

# ============================================================
# Convert Categorical Columns to Category Datatype
# ============================================================

categorical_columns = df.select_dtypes(

    include=["object"]

).columns

for column in categorical_columns:

    df[column] = df[column].astype("category")

print("\n")

print("="*70)
print("Categorical Columns Converted")
print("="*70)

print(categorical_columns.tolist())

# ============================================================
# Feature Matrix and Target Variable
# ============================================================

X = df.drop(

    "Item_Outlet_Sales",

    axis=1

)

y = df["Item_Outlet_Sales"]

print("\n")

print("="*70)
print("Feature Matrix and Target Variable")
print("="*70)

print("Features Shape :", X.shape)

print("Target Shape   :", y.shape)

# ============================================================
# CatBoost Categorical Feature Indices
# ============================================================

cat_features = [

    X.columns.get_loc(column)

    for column in categorical_columns

]

print("\n")

print("="*70)
print("Categorical Feature Indices")
print("="*70)

print(cat_features)

# ============================================================
# Display Features
# ============================================================

print("\n")

print("="*70)
print("First Five Rows of Features")
print("="*70)

display(X.head())

# ============================================================
# Display Target
# ============================================================

print("\n")

print("="*70)
print("First Five Target Values")
print("="*70)

display(y.head())

# ============================================================
# Final Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Final Dataset Summary")
print("="*70)

print("Total Samples      :", len(df))

print("Total Features     :", X.shape[1])

print("Numerical Features :", len(numerical_columns)-1)

print("Categorical Features :", len(categorical_columns))

print("\nDataset is Ready for CatBoost Training.")

print("="*70)

# ============================================================
#  Model Building & Training
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor

import numpy as np

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\n")

print("="*70)
print("Train Test Split")
print("="*70)

print("Training Features :", X_train.shape)

print("Testing Features  :", X_test.shape)

print("Training Labels   :", y_train.shape)

print("Testing Labels    :", y_test.shape)

# ============================================================
# Create CatBoost Model
# ============================================================

cat_model = CatBoostRegressor(

    iterations=200,

    learning_rate=0.1,

    depth=6,

    loss_function="RMSE",

    random_seed=42,

    verbose=0

)

# ============================================================
# Train Model
# ============================================================

cat_model.fit(

    X_train,

    y_train,

    cat_features=cat_features

)

print("\n")

print("="*70)
print("CatBoost Model Trained Successfully")
print("="*70)

# ============================================================
# Make Predictions
# ============================================================

predictions = cat_model.predict(

    X_test

)

# ============================================================
# Mean Absolute Error
# ============================================================

mae = mean_absolute_error(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("Mean Absolute Error")
print("="*70)

print("MAE :", round(mae,2))

# ============================================================
# Mean Squared Error
# ============================================================

mse = mean_squared_error(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("Mean Squared Error")
print("="*70)

print("MSE :", round(mse,2))

# ============================================================
# Root Mean Squared Error
# ============================================================

rmse = np.sqrt(mse)

print("\n")

print("="*70)
print("Root Mean Squared Error")
print("="*70)

print("RMSE :", round(rmse,2))

# ============================================================
# R2 Score
# ============================================================

r2 = r2_score(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("R2 Score")
print("="*70)

print("R2 Score :", round(r2,4))

# ============================================================
# Cross Validation
# ============================================================

cv_scores = cross_val_score(

    cat_model,

    X,

    y,

    cv=2,

    scoring="r2"

)

print("\n")

print("="*70)
print("Cross Validation")
print("="*70)

print("Scores :", cv_scores)

print("Average R2 Score :", round(cv_scores.mean(),4))

# ============================================================
# Hyperparameter Tuning
# ============================================================

parameters = {

    "iterations":[200],

    "learning_rate":[0.1],

    "depth":[6]

}

grid_search = GridSearchCV(

    estimator=cat_model,

    param_grid=parameters,

    cv=2,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1

)

grid_search.fit(

    X_train,

    y_train,

    cat_features=cat_features

)

print("\n")

print("="*70)
print("Best Parameters")
print("="*70)

print(grid_search.best_params_)

# ============================================================
# Best Model
# ============================================================

best_model = grid_search.best_estimator_

best_predictions = best_model.predict(

    X_test

)

# ============================================================
# Best Model Evaluation
# ============================================================

best_mae = mean_absolute_error(

    y_test,

    best_predictions

)

best_mse = mean_squared_error(

    y_test,

    best_predictions

)

best_rmse = np.sqrt(best_mse)

best_r2 = r2_score(

    y_test,

    best_predictions

)

print("\n")

print("="*70)
print("Best CatBoost Model Performance")
print("="*70)

print("MAE      :", round(best_mae,2))

print("MSE      :", round(best_mse,2))

print("RMSE     :", round(best_rmse,2))

print("R2 Score :", round(best_r2,4))

# ============================================================
# Feature Importance
# ============================================================

feature_importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":best_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)
print("Feature Importance")
print("="*70)

display(feature_importance)

# ============================================================
# Feature Importance Graph
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=feature_importance,

    x="Importance",

    y="Feature"

)

plt.title("CatBoost Feature Importance")

plt.show()

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("CatBoost Model Summary")
print("="*70)

print("Training Samples :", X_train.shape[0])

print("Testing Samples  :", X_test.shape[0])

print("MAE              :", round(best_mae,2))

print("MSE              :", round(best_mse,2))

print("RMSE             :", round(best_rmse,2))

print("R2 Score         :", round(best_r2,4))

print("Cross Validation :", round(cv_scores.mean(),4))

print("\n")

print("Model Status : Ready for Evaluation")

print("="*70)
# ============================================================
#  Model Evaluation & Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Actual vs Predicted Values
# ============================================================

comparison = pd.DataFrame({

    "Actual Sales": y_test.values,

    "Predicted Sales": best_predictions

})

print("\n")

print("="*70)
print("Actual vs Predicted Sales")
print("="*70)

display(comparison.head(20))

# ============================================================
# Scatter Plot
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,

    best_predictions,

    alpha=0.7

)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# ============================================================
# Residual Calculation
# ============================================================

residuals = y_test - best_predictions

print("\n")

print("="*70)
print("Residual Statistics")
print("="*70)

print(residuals.describe())

# ============================================================
# Residual Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    residuals,

    bins=30,

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

    best_predictions,

    residuals,

    alpha=0.7

)

plt.axhline(

    y=0,

    color="red",

    linestyle="--"

)

plt.xlabel("Predicted Sales")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.show()

# ============================================================
# Predict Sales for New Product
# ============================================================

new_product = X.iloc[[0]].copy()

predicted_sales = best_model.predict(

    new_product

)

print("\n")

print("="*70)
print("Predicted Sales for New Product")
print("="*70)

print("Predicted Sales :", round(predicted_sales[0],2))

# ============================================================
# Save Model
# ============================================================

joblib.dump(

    best_model,

    "sales_prediction_catboost_model.joblib"

)

print("\n")

print("="*70)
print("Model Saved Successfully")
print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = joblib.load(

    "sales_prediction_catboost_model.joblib"

)

print("\n")

print("="*70)
print("Saved Model Loaded Successfully")
print("="*70)

# ============================================================
# Prediction Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    new_product

)

print("\n")

print("="*70)
print("Prediction Using Loaded Model")
print("="*70)

print("Predicted Sales :", round(loaded_prediction[0],2))

# ============================================================
# Top 10 Important Features
# ============================================================

top_features = feature_importance.head(10)

print("\n")

print("="*70)
print("Top 10 Important Features")
print("="*70)

display(top_features)

# ============================================================
# Top 10 Feature Importance Graph
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=top_features,

    x="Importance",

    y="Feature"

)

plt.title("Top 10 Important Features")

plt.show()

# ============================================================
# Prediction Error
# ============================================================

comparison["Prediction Error"] = (

    comparison["Actual Sales"]

    - comparison["Predicted Sales"]

)

print("\n")

print("="*70)
print("Prediction Error")
print("="*70)

display(comparison.head(20))

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Sales Prediction using CatBoost")
print("="*70)

print("Algorithm Used        : CatBoost Regressor")

print("Training Samples      :", X_train.shape[0])

print("Testing Samples       :", X_test.shape[0])

print("Total Features        :", X.shape[1])

print("MAE                   :", round(best_mae,2))

print("MSE                   :", round(best_mse,2))

print("RMSE                  :", round(best_rmse,2))

print("R2 Score              :", round(best_r2,4))

print("Cross Validation      :", round(cv_scores.mean(),4))

print("Model File            : sales_prediction_catboost_model.joblib")

print("Deployment Status     : Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Sales Prediction using CatBoost Completed Successfully!")

print("="*70)
