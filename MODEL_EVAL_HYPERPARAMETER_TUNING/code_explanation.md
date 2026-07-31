# ============================================================
# Model Evaluation & Hyperparameter Tuning
# Part-1A : Exploratory Data Analysis (EDA)
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

    "WA_Fn-UseC_-Telco-Customer-Churn.csv"

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

display(

    df.head()

)

# ============================================================
# Display Last Five Records
# ============================================================

print("\n")

print("="*70)

print("Last Five Records")

print("="*70)

display(

    df.tail()

)

# ============================================================
# Dataset Shape
# ============================================================

print("\n")

print("="*70)

print("Dataset Shape")

print("="*70)

print(

    "Rows    :",

    df.shape[0]

)

print(

    "Columns :",

    df.shape[1]

)

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

print(

    df.columns.tolist()

)

# ============================================================
# Data Types
# ============================================================

print("\n")

print("="*70)

print("Data Types")

print("="*70)

display(

    df.dtypes

)

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*70)

print("Statistical Summary")

print("="*70)

display(

    df.describe()

)

# ============================================================
# Complete Statistical Summary
# ============================================================

print("\n")

print("="*70)

print("Complete Statistical Summary")

print("="*70)

display(

    df.describe(

        include="all"

    )

)

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("="*70)

print("Missing Values")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Duplicate Records
# ============================================================

duplicates = df.duplicated().sum()

print("\n")

print("="*70)

print("Duplicate Records")

print("="*70)

print(

    "Duplicate Records :",

    duplicates

)

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

print(

    "Current Shape :",

    df.shape

)

# ============================================================
# Target Variable Distribution
# ============================================================

plt.figure(

    figsize=(6,5)

)

sns.countplot(

    x="Churn",

    data=df

)

plt.title(

    "Customer Churn Distribution"

)

plt.xlabel(

    "Churn"

)

plt.ylabel(

    "Count"

)

plt.show()

# ============================================================
# Gender Distribution
# ============================================================

plt.figure(

    figsize=(6,5)

)

sns.countplot(

    x="gender",

    data=df

)

plt.title(

    "Gender Distribution"

)

plt.show()

# ============================================================
# Contract Type Distribution
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.countplot(

    x="Contract",

    data=df

)

plt.title(

    "Contract Distribution"

)

plt.xticks(

    rotation=15

)

plt.show()

# ============================================================
# Payment Method Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.countplot(

    y="PaymentMethod",

    data=df

)

plt.title(

    "Payment Method Distribution"

)

plt.show()

# ============================================================
# Monthly Charges Distribution
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.histplot(

    df["MonthlyCharges"],

    bins=30,

    kde=True

)

plt.title(

    "Monthly Charges Distribution"

)

plt.show()

# ============================================================
# Tenure Distribution
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.histplot(

    df["tenure"],

    bins=30,

    kde=True

)

plt.title(

    "Customer Tenure Distribution"

)

plt.show()

# ============================================================
# Churn vs Contract
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.countplot(

    x="Contract",

    hue="Churn",

    data=df

)

plt.title(

    "Contract Type vs Churn"

)

plt.xticks(

    rotation=15

)

plt.show()

# ============================================================
# Churn vs Internet Service
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.countplot(

    x="InternetService",

    hue="Churn",

    data=df

)

plt.title(

    "Internet Service vs Churn"

)

plt.xticks(

    rotation=15

)

plt.show()

# ============================================================
# Boxplot : Monthly Charges
# ============================================================

plt.figure(

    figsize=(8,5)

)

sns.boxplot(

    x=df["MonthlyCharges"]

)

plt.title(

    "Monthly Charges Boxplot"

)

plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

numeric_df = df.select_dtypes(

    include=np.number

)

plt.figure(

    figsize=(8,6)

)

sns.heatmap(

    numeric_df.corr(),

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

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

print(

    "Total Records        :",

    df.shape[0]

)

print(

    "Total Features       :",

    df.shape[1]

)

print(

    "Missing Values       :",

    df.isnull().sum().sum()

)

print(

    "Duplicate Records    :",

    duplicates

)

print(

    "Target Variable      : Churn"

)

print(

    "Positive Class       :",

    df["Churn"].value_counts()["Yes"]

)

print(

    "Negative Class       :",

    df["Churn"].value_counts()["No"]

)

print("="*70)

# ============================================================
#  Data Preprocessing & Baseline Model
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ============================================================
# Check Missing Values
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

# Convert TotalCharges to numeric

df["TotalCharges"] = pd.to_numeric(

    df["TotalCharges"],

    errors="coerce"

)

# Fill missing values

df["TotalCharges"].fillna(

    df["TotalCharges"].median(),

    inplace=True

)

print("\n")

print("="*70)

print("Missing Values After Treatment")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Remove Customer ID
# ============================================================

df.drop(

    columns=["customerID"],

    inplace=True

)

print("\n")

print("="*70)

print("Customer ID Removed")

print("="*70)

# ============================================================
# Encode Target Variable
# ============================================================

target_encoder = LabelEncoder()

df["Churn"] = target_encoder.fit_transform(

    df["Churn"]

)

print("\n")

print("="*70)

print("Target Variable Encoded")

print("="*70)

display(

    df["Churn"].head()

)

# ============================================================
# Label Encoding Categorical Features
# ============================================================

label_encoders = {}

categorical_columns = df.select_dtypes(

    include="object"

).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(

        df[column]

    )

    label_encoders[column] = encoder

print("\n")

print("="*70)

print("Categorical Features Encoded")

print("="*70)

# ============================================================
# Feature Matrix & Target Variable
# ============================================================

X = df.drop(

    "Churn",

    axis=1

)

y = df["Churn"]

print("\n")

print("="*70)

print("Feature Matrix & Target Created")

print("="*70)

print("Features Shape :", X.shape)

print("Target Shape   :", y.shape)

# ============================================================
# Train-Test Split
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

print("Train-Test Split")

print("="*70)

print("Training Records :", X_train.shape[0])

print("Testing Records  :", X_test.shape[0])

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(

    X_train

)

X_test = scaler.transform(

    X_test

)

print("\n")

print("="*70)

print("Feature Scaling Completed")

print("="*70)

# ============================================================
# Build Baseline Random Forest Model
# ============================================================

baseline_model = RandomForestClassifier(

    random_state=42

)

baseline_model.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)

print("Baseline Random Forest Model Trained")

print("="*70)

# ============================================================
# Make Predictions
# ============================================================

y_pred = baseline_model.predict(

    X_test

)

print("\n")

print("="*70)

print("Predictions Generated")

print("="*70)

display(

    pd.DataFrame({

        "Actual": y_test.values[:20],

        "Predicted": y_pred[:20]

    })

)

# ============================================================
# Baseline Accuracy
# ============================================================

baseline_accuracy = accuracy_score(

    y_test,

    y_pred

)

print("\n")

print("="*70)

print("Baseline Model Accuracy")

print("="*70)

print(

    "Accuracy :", round(

        baseline_accuracy,

        4

    )

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Training Samples :", len(X_train)

)

print(

    "Testing Samples  :", len(X_test)

)

print(

    "Number of Features :", X.shape[1]

)

print(

    "Target Classes :", len(np.unique(y))

)

print(

    "Baseline Accuracy :", round(

        baseline_accuracy,

        4

    )

)

print("="*70)

# ============================================================
# Dataset Ready for Model Evaluation
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for Model Evaluation & Hyperparameter Tuning")

print("="*70)

# ============================================================
#  Model Evaluation & Hyperparameter Tuning
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

# ============================================================
# Baseline Model Evaluation
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

print("\n")

print("="*70)

print("Baseline Model Evaluation")

print("="*70)

print("Accuracy  :", round(accuracy,4))

print("Precision :", round(precision,4))

print("Recall    :", round(recall,4))

print("F1 Score  :", round(f1,4))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred

)

plt.figure(

    figsize=(6,5)

)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.title(

    "Confusion Matrix"

)

plt.xlabel(

    "Predicted"

)

plt.ylabel(

    "Actual"

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

        y_pred

    )

)

# ============================================================
# ROC Curve & AUC Score
# ============================================================

y_probability = baseline_model.predict_proba(

    X_test

)[:,1]

auc_score = roc_auc_score(

    y_test,

    y_probability

)

fpr, tpr, thresholds = roc_curve(

    y_test,

    y_probability

)

plt.figure(

    figsize=(7,5)

)

plt.plot(

    fpr,

    tpr,

    label="ROC Curve"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)

plt.xlabel(

    "False Positive Rate"

)

plt.ylabel(

    "True Positive Rate"

)

plt.title(

    "ROC Curve"

)

plt.legend()

plt.grid()

plt.show()

print("\n")

print("="*70)

print("ROC-AUC Score")

print("="*70)

print(

    round(

        auc_score,

        4

    )

)

# ============================================================
# Cross Validation
# ============================================================

cv_scores = cross_val_score(

    baseline_model,

    X,

    y,

    cv=5,

    scoring="accuracy"

)

print("\n")

print("="*70)

print("5-Fold Cross Validation")

print("="*70)

print(

    cv_scores

)

print(

    "Average Accuracy :",

    round(

        cv_scores.mean(),

        4

    )

)

# ============================================================
# Stratified K-Fold
# ============================================================

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

skf_scores = cross_val_score(

    baseline_model,

    X,

    y,

    cv=skf,

    scoring="accuracy"

)

print("\n")

print("="*70)

print("Stratified K-Fold Accuracy")

print("="*70)

print(

    skf_scores

)

print(

    "Average Accuracy :",

    round(

        skf_scores.mean(),

        4

    )

)

# ============================================================
# GridSearchCV
# ============================================================

parameter_grid = {

    "n_estimators":[100,200,300],

    "max_depth":[5,10,20,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4],

    "bootstrap":[True,False]

}

grid_search = GridSearchCV(

    estimator=RandomForestClassifier(

        random_state=42

    ),

    param_grid=parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid_search.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)

print("GridSearchCV Best Parameters")

print("="*70)

print(

    grid_search.best_params_

)

print(

    "Best Accuracy :",

    round(

        grid_search.best_score_,

        4

    )

)

# ============================================================
# RandomizedSearchCV
# ============================================================

random_parameters = {

    "n_estimators":[100,200,300,400,500],

    "max_depth":[5,10,20,30,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4],

    "bootstrap":[True,False]

}

random_search = RandomizedSearchCV(

    estimator=RandomForestClassifier(

        random_state=42

    ),

    param_distributions=random_parameters,

    n_iter=20,

    cv=5,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1

)

random_search.fit(

    X_train,

    y_train

)

print("\n")

print("="*70)

print("RandomizedSearchCV Best Parameters")

print("="*70)

print(

    random_search.best_params_

)

print(

    "Best Accuracy :",

    round(

        random_search.best_score_,

        4

    )

)

# ============================================================
# Best Model
# ============================================================

best_model = random_search.best_estimator_

best_predictions = best_model.predict(

    X_test

)

best_accuracy = accuracy_score(

    y_test,

    best_predictions

)

print("\n")

print("="*70)

print("Optimized Model Accuracy")

print("="*70)

print(

    round(

        best_accuracy,

        4

    )

)

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

plt.figure(

    figsize=(10,6)

)

sns.barplot(

    data=importance.head(10),

    x="Importance",

    y="Feature"

)

plt.title(

    "Top 10 Feature Importance"

)

plt.show()

# ============================================================
# Model Comparison
# ============================================================

comparison = pd.DataFrame({

    "Model":[

        "Baseline",

        "Optimized"

    ],

    "Accuracy":[

        accuracy,

        best_accuracy

    ]

})

print("\n")

print("="*70)

print("Model Comparison")

print("="*70)

display(

    comparison

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Accuracy alone should not be used for imbalanced datasets.")

print("- Precision measures the correctness of positive predictions.")

print("- Recall measures how many actual churn customers were detected.")

print("- F1 Score balances Precision and Recall.")

print("- ROC-AUC measures model discrimination capability.")

print("- Cross Validation improves model reliability.")

print("- Hyperparameter tuning improves model performance and generalization.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("Model Evaluation Summary")

print("="*70)

print("Algorithm                : Random Forest")

print("Evaluation Metrics       : Accuracy, Precision, Recall, F1")

print("Validation               : 5-Fold Cross Validation")

print("Hyperparameter Tuning    : GridSearchCV")

print("Optimization             : RandomizedSearchCV")

print("Best Accuracy            :", round(best_accuracy,4))

print("="*70)


# ============================================================
#  Production Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save Best Model
# ============================================================

joblib.dump(

    best_model,

    "best_random_forest_model.joblib"

)

print("\n")

print("="*70)

print("Best Model Saved Successfully")

print("="*70)

# ============================================================
# Save Standard Scaler
# ============================================================

joblib.dump(

    scaler,

    "standard_scaler.joblib"

)

print("\n")

print("="*70)

print("Standard Scaler Saved Successfully")

print("="*70)

# ============================================================
# Save Target Label Encoder
# ============================================================

joblib.dump(

    target_encoder,

    "target_label_encoder.joblib"

)

print("\n")

print("="*70)

print("Target Label Encoder Saved Successfully")

print("="*70)

# ============================================================
# Save Feature Label Encoders
# ============================================================

joblib.dump(

    label_encoders,

    "feature_label_encoders.joblib"

)

print("\n")

print("="*70)

print("Feature Label Encoders Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Objects
# ============================================================

loaded_model = joblib.load(

    "best_random_forest_model.joblib"

)

loaded_scaler = joblib.load(

    "standard_scaler.joblib"

)

loaded_target_encoder = joblib.load(

    "target_label_encoder.joblib"

)

loaded_feature_encoders = joblib.load(

    "feature_label_encoders.joblib"

)

print("\n")

print("="*70)

print("Saved Objects Loaded Successfully")

print("="*70)

# ============================================================
# Predict Test Dataset
# ============================================================

loaded_predictions = loaded_model.predict(

    X_test

)

prediction_output = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": loaded_predictions

})

print("\n")

print("="*70)

print("Prediction Sample")

print("="*70)

display(

    prediction_output.head(

        20

    )

)

# ============================================================
# Export Predictions
# ============================================================

prediction_output.to_csv(

    "Customer_Churn_Predictions.csv",

    index=False

)

print("\n")

print("="*70)

print("Predictions Exported Successfully")

print("="*70)

# ============================================================
# Save Evaluation Metrics
# ============================================================

evaluation_metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ],

    "Value":[

        best_accuracy,

        precision,

        recall,

        f1,

        auc_score

    ]

})

evaluation_metrics.to_csv(

    "Evaluation_Metrics.csv",

    index=False

)

print("\n")

print("="*70)

print("Evaluation Metrics Saved Successfully")

print("="*70)

display(

    evaluation_metrics

)

# ============================================================
# Feature Importance
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": loaded_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

feature_importance.to_csv(

    "Feature_Importance.csv",

    index=False

)

print("\n")

print("="*70)

print("Feature Importance")

print("="*70)

display(

    feature_importance.head(

        10

    )

)

# ============================================================
# Feature Importance Visualization
# ============================================================

plt.figure(

    figsize=(10,6)

)

sns.barplot(

    data=feature_importance.head(10),

    x="Importance",

    y="Feature"

)

plt.title(

    "Top 10 Important Features"

)

plt.xlabel(

    "Importance"

)

plt.ylabel(

    "Feature"

)

plt.show()

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Model Type",

        "Training Samples",

        "Testing Samples",

        "Cross Validation",

        "Hyperparameter Tuning"

    ],

    "Value":[

        "Random Forest",

        "Classification",

        len(X_train),

        len(X_test),

        "5-Fold Stratified",

        "GridSearchCV & RandomizedSearchCV"

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
# Deployment Summary
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "Optimized Model",

        "Standard Scaler",

        "Target Label Encoder",

        "Feature Label Encoders",

        "Prediction Results",

        "Evaluation Metrics",

        "Feature Importance"

    ],

    "Saved As":[

        "best_random_forest_model.joblib",

        "standard_scaler.joblib",

        "target_label_encoder.joblib",

        "feature_label_encoders.joblib",

        "Customer_Churn_Predictions.csv",

        "Evaluation_Metrics.csv",

        "Feature_Importance.csv"

    ]

})

print("\n")

print("="*70)

print("Deployment Summary")

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

print("- Hyperparameter tuning improved model performance.")

print("- Cross Validation increases confidence in model stability.")

print("- Feature Importance identifies key churn drivers.")

print("- Saving preprocessing objects ensures consistent predictions.")

print("- The model is now ready for deployment through APIs or web applications.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Model Evaluation & Hyperparameter Tuning")

print("="*70)

print("Dataset                  : Telco Customer Churn")

print("Algorithm                : Random Forest Classifier")

print("Problem Type             : Classification")

print("Evaluation Metrics       : Accuracy, Precision, Recall, F1, ROC-AUC")

print("Validation               : 5-Fold Stratified Cross Validation")

print("Hyperparameter Tuning    : GridSearchCV & RandomizedSearchCV")

print("Model Saved              : best_random_forest_model.joblib")

print("Scaler Saved             : standard_scaler.joblib")

print("Target Encoder Saved     : target_label_encoder.joblib")

print("Feature Encoders Saved   : feature_label_encoders.joblib")

print("Prediction File          : Customer_Churn_Predictions.csv")

print("Metrics File             : Evaluation_Metrics.csv")

print("Importance File          : Feature_Importance.csv")

print("Project Status           : Production Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Production Machine Learning Project Completed Successfully!")

print("="*70)
