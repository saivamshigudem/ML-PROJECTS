# ============================================================
# Insurance Risk Prediction using Gradient Boosting
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

# Display All Columns
pd.set_option("display.max_columns", None)

# ============================================================
# Load Training Dataset
# ============================================================

df = pd.read_csv("gd-train.csv")

print("="*70)
print("Insurance Risk Prediction Dataset Loaded Successfully")
print("="*70)

# ============================================================
# Display First Five Rows
# ============================================================

print("\nFirst Five Records\n")

display(df.head())

# ============================================================
# Display Last Five Rows
# ============================================================

print("\nLast Five Records\n")

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
# Column Names
# ============================================================

print("\n")

print("="*70)
print("Column Names")
print("="*70)

print(df.columns)

# ============================================================
# Dataset Information
# ============================================================

print("\n")

print("="*70)
print("Dataset Information")
print("="*70)

df.info()

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*70)
print("Statistical Summary")
print("="*70)

display(df.describe())

# ============================================================
# Statistical Summary Including Categorical Columns
# ============================================================

print("\n")

print("="*70)
print("Complete Statistical Summary")
print("="*70)

display(df.describe(include="all"))

# ============================================================
# Check Missing Values
# ============================================================

print("\n")

print("="*70)
print("Missing Values")
print("="*70)

print(df.isnull().sum())

print("\nTotal Missing Values :", df.isnull().sum().sum())

# ============================================================
# Check Duplicate Records
# ============================================================

print("\n")

print("="*70)
print("Duplicate Records")
print("="*70)

duplicates = df.duplicated().sum()

print("Duplicate Records :", duplicates)

# Remove Duplicates

df = df.drop_duplicates()

print("\nShape After Removing Duplicates")

print(df.shape)

# ============================================================
# Number of Unique Values
# ============================================================

print("\n")

print("="*70)
print("Unique Values")
print("="*70)

display(df.nunique())

# ============================================================
# Separate Numerical & Categorical Columns
# ============================================================

numerical_columns = df.select_dtypes(include=np.number).columns

categorical_columns = df.select_dtypes(include="object").columns

print("\n")

print("="*70)
print("Numerical Columns")
print("="*70)

print(numerical_columns)

print("\n")

print("="*70)
print("Categorical Columns")
print("="*70)

print(categorical_columns)

# ============================================================
# Value Counts of Categorical Columns
# ============================================================

for column in categorical_columns:

    print("\n")

    print("="*70)

    print(column)

    print("="*70)

    print(df[column].value_counts())

# ============================================================
# Target Variable Distribution
# ============================================================

plt.figure(figsize=(6,4))

sns.countplot(

    x="Response",

    data=df,

    palette="Set2"

)

plt.title("Response Distribution")

plt.xlabel("Response")

plt.ylabel("Count")

plt.show()

# ============================================================
# Numerical Feature Distribution
# ============================================================

df[numerical_columns].hist(

    figsize=(18,12),

    bins=20

)

plt.tight_layout()

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

plt.figure(figsize=(12,9))

sns.heatmap(

    df.corr(numeric_only=True),

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# Boxplots for Numerical Features
# ============================================================

for column in numerical_columns:

    plt.figure(figsize=(6,3))

    sns.boxplot(

        x=df[column],

        color="skyblue"

    )

    plt.title(column)

    plt.show()



# ============================================================
# Label Encoding
# ============================================================
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for column in categorical_columns:

    df[column] = encoder.fit_transform(df[column])

print("\n")

print("="*70)
print("Label Encoding Completed")
print("="*70)

display(df.head())
# ============================================================
# Separate Features and Target Variable
# ============================================================

X = df.drop(

    "Response",

    axis=1

)

y = df["Response"]

print("\n")

print("="*70)
print("Feature Matrix and Target Variable")
print("="*70)

print("Features Shape :", X.shape)

print("Target Shape   :", y.shape)

# ============================================================
# Display First Five Rows of Features
# ============================================================

print("\nFeatures")

display(X.head())

# ============================================================
# Display First Five Target Values
# ============================================================

print("\nTarget")

display(y.head())

# ============================================================
# Final Dataset Summary
# ============================================================

print("\n")

print("="*70)
print("Final Dataset Summary")
print("="*70)

print(f"Total Samples  : {len(df)}")

print(f"Total Features : {X.shape[1]}")

print(f"Target Column  : Response")

print("\nDataset is Ready for Model Building.")

print("="*70)

print("\nPart-1 Completed Successfully!")

# ============================================================
#  Model Building & Training
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

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
# Create Gradient Boosting Model
# ============================================================

gb_model = GradientBoostingClassifier(

    random_state=42

)

# ============================================================
# Train Model
# ============================================================

gb_model.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)
print("Gradient Boosting Model Trained Successfully")
print("="*70)

# ============================================================
# Make Predictions
# ============================================================

predictions = gb_model.predict(

    X_test

)

# ============================================================
# Model Accuracy
# ============================================================

accuracy = accuracy_score(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("Accuracy")
print("="*70)

print("Accuracy :", round(accuracy,4))

# ============================================================
# Precision
# ============================================================

precision = precision_score(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("Precision")
print("="*70)

print("Precision :", round(precision,4))

# ============================================================
# Recall
# ============================================================

recall = recall_score(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("Recall")
print("="*70)

print("Recall :", round(recall,4))

# ============================================================
# F1 Score
# ============================================================

f1 = f1_score(

    y_test,

    predictions

)

print("\n")

print("="*70)
print("F1 Score")
print("="*70)

print("F1 Score :", round(f1,4))

# ============================================================
# Cross Validation
# ============================================================

cv_scores = cross_val_score(

    gb_model,

    X,

    y,

    cv=5,

    scoring="accuracy"

)

print("\n")

print("="*70)
print("Cross Validation")
print("="*70)

print("Scores :", cv_scores)

print("Average Accuracy :", round(cv_scores.mean(),4))

# ============================================================
# Hyperparameter Tuning
# ============================================================

parameters = {

    "n_estimators":[100,150],

    "learning_rate":[0.05,0.1],

    "max_depth":[3,5],

    "min_samples_split":[2,5],

    "min_samples_leaf":[1,2]

}

grid_search = GridSearchCV(

    estimator=GradientBoostingClassifier(random_state=42),

    param_grid=parameters,

    cv=3,

    scoring="accuracy",

    n_jobs=-1

)

grid_search.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)
print("Best Parameters")
print("="*70)

print(grid_search.best_params_)

# ============================================================
# Best Gradient Boosting Model
# ============================================================

best_model = grid_search.best_estimator_

best_predictions = best_model.predict(

    X_test

)

# ============================================================
# Evaluate Best Model
# ============================================================

best_accuracy = accuracy_score(

    y_test,

    best_predictions

)

best_precision = precision_score(

    y_test,

    best_predictions

)

best_recall = recall_score(

    y_test,

    best_predictions

)

best_f1 = f1_score(

    y_test,

    best_predictions

)

print("\n")

print("="*70)
print("Best Gradient Boosting Model Performance")
print("="*70)

print("Accuracy  :", round(best_accuracy,4))

print("Precision :", round(best_precision,4))

print("Recall    :", round(best_recall,4))

print("F1 Score  :", round(best_f1,4))

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

print(feature_importance)

# ============================================================
# Feature Importance Graph
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    data=feature_importance,

    x="Importance",

    y="Feature",

    palette="viridis"

)

plt.title("Gradient Boosting Feature Importance")

plt.show()

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)
print("Gradient Boosting Model Summary")
print("="*70)

print(f"Training Samples : {X_train.shape[0]}")

print(f"Testing Samples  : {X_test.shape[0]}")

print(f"Accuracy         : {best_accuracy:.4f}")

print(f"Precision        : {best_precision:.4f}")

print(f"Recall           : {best_recall:.4f}")

print(f"F1 Score         : {best_f1:.4f}")

print(f"Cross Validation : {cv_scores.mean():.4f}")

print("\n")

print("Model Status : Ready for Evaluation")

print("="*70)

# ============================================================
#  Model Evaluation & Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from joblib import dump
from joblib import load

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_test,

    best_predictions

)

print("\n")

print("="*70)
print("Confusion Matrix")
print("="*70)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

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

        y_test,

        best_predictions

    )

)

# ============================================================
# ROC Curve
# ============================================================

prediction_probability = best_model.predict_proba(

    X_test

)[:,1]

fpr, tpr, threshold = roc_curve(

    y_test,

    prediction_probability

)

auc_score = roc_auc_score(

    y_test,

    prediction_probability

)

plt.figure(figsize=(7,5))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {auc_score:.4f}"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    color="red"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("\n")

print("="*70)
print("ROC-AUC Score")
print("="*70)

print("AUC Score :", round(auc_score,4))

# ============================================================
# Predict New Customer
# ============================================================

new_customer = pd.DataFrame({

    "id":[381110],

    "Gender":[1],

    "Age":[35],

    "Driving_License":[1],

    "Region_Code":[28],

    "Previously_Insured":[0],

    "Vehicle_Age":[2],

    "Vehicle_Damage":[1],

    "Annual_Premium":[35000],

    "Policy_Sales_Channel":[26],

    "Vintage":[180]

})

prediction = best_model.predict(

    new_customer

)

print("\n")

print("="*70)
print("Prediction for New Customer")
print("="*70)

if prediction[0] == 1:

    print("Customer is Interested in Vehicle Insurance")

else:

    print("Customer is Not Interested in Vehicle Insurance")

# ============================================================
# Prediction Probability
# ============================================================

probability = best_model.predict_proba(

    new_customer

)

print("\nPrediction Probability")

print(probability)

# ============================================================
# Save Model
# ============================================================

dump(

    best_model,

    "gradient_boosting_insurance_model.joblib"

)

print("\n")

print("="*70)
print("Model Saved Successfully")
print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = load(

    "gradient_boosting_insurance_model.joblib"

)

print("\n")

print("="*70)
print("Saved Model Loaded Successfully")
print("="*70)

# ============================================================
# Predict Using Loaded Model
# ============================================================

loaded_prediction = loaded_model.predict(

    new_customer

)

print("\n")

print("="*70)
print("Prediction Using Loaded Model")
print("="*70)

if loaded_prediction[0] == 1:

    print("Customer is Interested in Vehicle Insurance")

else:

    print("Customer is Not Interested in Vehicle Insurance")

# ============================================================
# Show First 10 Actual vs Predicted Values
# ============================================================

comparison = pd.DataFrame({

    "Actual":y_test.iloc[:10].values,

    "Predicted":best_predictions[:10]

})

print("\n")

print("="*70)
print("Actual vs Predicted")
print("="*70)

print(comparison)

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)
print("Feature Importance")
print("="*70)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(

    data=importance,

    x="Importance",

    y="Feature",

    palette="viridis"

)

plt.title("Gradient Boosting Feature Importance")

plt.show()

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("Insurance Risk Prediction using Gradient Boosting")
print("="*70)

print(f"Training Samples : {X_train.shape[0]}")

print(f"Testing Samples  : {X_test.shape[0]}")

print(f"Total Features   : {X.shape[1]}")

print(f"Accuracy         : {best_accuracy:.4f}")

print(f"Precision        : {best_precision:.4f}")

print(f"Recall           : {best_recall:.4f}")

print(f"F1 Score         : {best_f1:.4f}")

print(f"ROC-AUC Score    : {auc_score:.4f}")

print(f"Cross Validation : {cv_scores.mean():.4f}")

print("\n")

print("Model Status : Ready for Deployment")

print("="*70)

print("\nProject Completed Successfully!")
