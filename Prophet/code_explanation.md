# ============================================================
# Business Forecasting using Prophet
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

    "Sample - Superstore.csv",

    encoding="latin1"

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
# Convert Order Date
# ============================================================

df["Order Date"] = pd.to_datetime(

    df["Order Date"]

)

print("\n")

print("="*70)

print("Order Date Converted Successfully")

print("="*70)

# ============================================================
# Sort Dataset by Order Date
# ============================================================

df = df.sort_values(

    by="Order Date"

)

print("\n")

print("="*70)

print("Dataset Sorted by Order Date")

print("="*70)

# ============================================================
# Sales Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    df["Sales"],

    bins=40,

    kde=True

)

plt.title(

    "Sales Distribution"

)

plt.xlabel(

    "Sales"

)

plt.show()

# ============================================================
# Sales Trend Over Time
# ============================================================

daily_sales = df.groupby(

    "Order Date"

)["Sales"].sum()

plt.figure(

    figsize=(15,6)

)

plt.plot(

    daily_sales

)

plt.title(

    "Daily Sales Trend"

)

plt.xlabel(

    "Order Date"

)

plt.ylabel(

    "Sales"

)

plt.grid()

plt.show()

# ============================================================
# Monthly Sales Trend
# ============================================================

monthly_sales = df.groupby(

    df["Order Date"].dt.to_period("M")

)["Sales"].sum()

monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(

    figsize=(15,6)

)

plt.plot(

    monthly_sales,

    marker="o"

)

plt.title(

    "Monthly Sales"

)

plt.xlabel(

    "Month"

)

plt.ylabel(

    "Sales"

)

plt.grid()

plt.show()

# ============================================================
# Yearly Sales Trend
# ============================================================

yearly_sales = df.groupby(

    df["Order Date"].dt.year

)["Sales"].sum()

plt.figure(

    figsize=(10,5)

)

yearly_sales.plot(

    kind="bar"

)

plt.title(

    "Yearly Sales"

)

plt.xlabel(

    "Year"

)

plt.ylabel(

    "Sales"

)

plt.show()

# ============================================================
# Top 10 States by Sales
# ============================================================

top_states = df.groupby(

    "State"

)["Sales"].sum().sort_values(

    ascending=False

).head(10)

plt.figure(

    figsize=(12,5)

)

top_states.plot(

    kind="bar"

)

plt.title(

    "Top 10 States by Sales"

)

plt.ylabel(

    "Sales"

)

plt.show()

# ============================================================
# Top 10 Categories by Sales
# ============================================================

category_sales = df.groupby(

    "Category"

)["Sales"].sum()

plt.figure(

    figsize=(8,5)

)

category_sales.plot(

    kind="bar"

)

plt.title(

    "Category-wise Sales"

)

plt.ylabel(

    "Sales"

)

plt.show()

# ============================================================
# Region-wise Sales
# ============================================================

region_sales = df.groupby(

    "Region"

)["Sales"].sum()

plt.figure(

    figsize=(7,7)

)

region_sales.plot(

    kind="pie",

    autopct="%1.1f%%"

)

plt.ylabel("")

plt.title(

    "Region-wise Sales Share"

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

    cmap="Blues",

    fmt=".2f"

)

plt.title(

    "Correlation Matrix"

)

plt.show()

# ============================================================
# Top 10 Products by Sales
# ============================================================

top_products = df.groupby(

    "Product Name"

)["Sales"].sum().sort_values(

    ascending=False

).head(10)

plt.figure(

    figsize=(12,5)

)

top_products.plot(

    kind="bar"

)

plt.title(

    "Top 10 Products by Sales"

)

plt.ylabel(

    "Sales"

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

    "Total Records       :",

    df.shape[0]

)

print(

    "Total Features      :",

    df.shape[1]

)

print(

    "Date Range          :",

    df["Order Date"].min(),

    "to",

    df["Order Date"].max()

)

print(

    "Total Sales         :",

    round(

        df["Sales"].sum(),

        2

    )

)

print(

    "Average Sales       :",

    round(

        df["Sales"].mean(),

        2

    )

)

print(

    "Maximum Sale        :",

    round(

        df["Sales"].max(),

        2

    )

)

print(

    "Minimum Sale        :",

    round(

        df["Sales"].min(),

        2

    )

)

print("="*70)

# ============================================================
# Data Preprocessing & Prophet Dataset Preparation
# ============================================================

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

df.ffill(inplace=True) 

df.bfill(inplace=True) 

print("\n")

print("="*70)

print("Missing Values After Treatment")

print("="*70)

display(

    df.isnull().sum()

)

# ============================================================
# Remove Duplicate Records
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
# Keep Required Columns
# ============================================================

prophet_df = df[[

    "Order Date",

    "Sales"

]].copy()

print("\n")

print("="*70)

print("Required Columns Selected")

print("="*70)

display(

    prophet_df.head()

)

# ============================================================
# Rename Columns for Prophet
# ============================================================

prophet_df.rename(

    columns={

        "Order Date":"ds",

        "Sales":"y"

    },

    inplace=True

)

print("\n")

print("="*70)

print("Columns Renamed Successfully")

print("="*70)

display(

    prophet_df.head()

)

# ============================================================
# Aggregate Daily Sales
# ============================================================

prophet_df = prophet_df.groupby(

    "ds",

    as_index=False

)["y"].sum()

print("\n")

print("="*70)

print("Daily Sales Aggregated")

print("="*70)

display(

    prophet_df.head()

)

# ============================================================
# Convert Date Format
# ============================================================

prophet_df["ds"] = pd.to_datetime(

    prophet_df["ds"]

)

print("\n")

print("="*70)

print("Date Format Converted")

print("="*70)

# ============================================================
# Sort Dataset
# ============================================================

prophet_df = prophet_df.sort_values(

    by="ds"

)

print("\n")

print("="*70)

print("Dataset Sorted Successfully")

print("="*70)

# ============================================================
# Reset Index
# ============================================================

prophet_df.reset_index(

    drop=True,

    inplace=True

)

print("\n")

print("="*70)

print("Index Reset Successfully")

print("="*70)

# ============================================================
# Prophet Dataset Information
# ============================================================

print("\n")

print("="*70)

print("Prophet Dataset Information")

print("="*70)

display(

    prophet_df.head()

)

display(

    prophet_df.tail()

)

# ============================================================
# Dataset Shape
# ============================================================

print("\n")

print("="*70)

print("Dataset Shape")

print("="*70)

print(

    "Rows :",

    prophet_df.shape[0]

)

print(

    "Columns :",

    prophet_df.shape[1]

)

# ============================================================
# Check Data Types
# ============================================================

print("\n")

print("="*70)

print("Data Types")

print("="*70)

display(

    prophet_df.dtypes

)

# ============================================================
# Check Missing Values Again
# ============================================================

print("\n")

print("="*70)

print("Final Missing Values")

print("="*70)

display(

    prophet_df.isnull().sum()

)

# ============================================================
# Time Series Visualization
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    prophet_df["ds"],

    prophet_df["y"]

)

plt.title(

    "Business Sales Time Series"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Sales"

)

plt.grid()

plt.show()

# ============================================================
# Rolling Mean
# ============================================================

prophet_df["Rolling_Mean"] = prophet_df["y"].rolling(

    window=30

).mean()

plt.figure(

    figsize=(15,6)

)

plt.plot(

    prophet_df["ds"],

    prophet_df["y"],

    label="Actual Sales"

)

plt.plot(

    prophet_df["ds"],

    prophet_df["Rolling_Mean"],

    label="30-Day Rolling Mean"

)

plt.legend()

plt.grid()

plt.title(

    "Rolling Mean of Sales"

)

plt.show()

# ============================================================
# Train-Test Split
# ============================================================

train_size = int(

    len(prophet_df) * 0.80

)

train = prophet_df.iloc[:train_size]

test = prophet_df.iloc[train_size:]

print("\n")

print("="*70)

print("Train-Test Split")

print("="*70)

print(

    "Training Records :",

    len(train)

)

print(

    "Testing Records :",

    len(test)

)

# ============================================================
# Train vs Test Visualization
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    train["ds"],

    train["y"],

    label="Training Data"

)

plt.plot(

    test["ds"],

    test["y"],

    label="Testing Data"

)

plt.legend()

plt.grid()

plt.title(

    "Train and Test Dataset"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Sales"

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

    len(prophet_df)

)

print(

    "Training Records     :",

    len(train)

)

print(

    "Testing Records      :",

    len(test)

)

print(

    "Start Date           :",

    prophet_df["ds"].min()

)

print(

    "End Date             :",

    prophet_df["ds"].max()

)

print(

    "Maximum Sales        :",

    round(

        prophet_df["y"].max(),

        2

    )

)

print(

    "Minimum Sales        :",

    round(

        prophet_df["y"].min(),

        2

    )

)

print(

    "Average Sales        :",

    round(

        prophet_df["y"].mean(),

        2

    )

)

print("="*70)

# ============================================================
# Prophet Dataset Ready
# ============================================================

print("\n")

print("="*70)

print("Dataset Ready for Prophet Model")

print("="*70)

# ============================================================
#  Prophet Model Building & Forecasting
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from prophet import Prophet

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import numpy as np

# ============================================================
# Build Prophet Model
# ============================================================

model = Prophet(

    yearly_seasonality=True,

    weekly_seasonality=True,

    daily_seasonality=False

)

print("\n")

print("="*70)

print("Prophet Model Created Successfully")

print("="*70)

# ============================================================
# Train Model
# ============================================================

model.fit(

    train

)

print("\n")

print("="*70)

print("Model Training Completed")

print("="*70)

# ============================================================
# Generate Future Dates
# ============================================================

future = model.make_future_dataframe(

    periods=len(test),

    freq="D"

)

print("\n")

print("="*70)

print("Future Dates Generated")

print("="*70)

display(

    future.tail()

)

# ============================================================
# Forecast Sales
# ============================================================

forecast = model.predict(

    future

)

print("\n")

print("="*70)

print("Forecast Generated Successfully")

print("="*70)

display(

    forecast[

        [

            "ds",

            "yhat",

            "yhat_lower",

            "yhat_upper"

        ]

    ].tail()

)

# ============================================================
# Extract Test Forecast
# ============================================================

predictions = forecast[

    [

        "ds",

        "yhat"

    ]

].tail(

    len(test)

)

predictions.reset_index(

    drop=True,

    inplace=True

)

test.reset_index(

    drop=True,

    inplace=True

)

# ============================================================
# Evaluation Metrics
# ============================================================

mae = mean_absolute_error(

    test["y"],

    predictions["yhat"]

)

mse = mean_squared_error(

    test["y"],

    predictions["yhat"]

)

rmse = np.sqrt(

    mse

)

print("\n")

print("="*70)

print("Evaluation Metrics")

print("="*70)

print(

    "MAE  :",

    round(

        mae,

        2

    )

)

print(

    "MSE  :",

    round(

        mse,

        2

    )

)

print(

    "RMSE :",

    round(

        rmse,

        2

    )

)

# ============================================================
# Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Date": test["ds"],

    "Actual Sales": test["y"],

    "Predicted Sales": predictions["yhat"]

})

print("\n")

print("="*70)

print("Actual vs Predicted")

print("="*70)

display(

    comparison.head(

        20

    )

)

# ============================================================
# Forecast Plot
# ============================================================

fig1 = model.plot(

    forecast

)

plt.title(

    "Business Sales Forecast"

)

plt.grid()

plt.show()

# ============================================================
# Prophet Components
# ============================================================

fig2 = model.plot_components(

    forecast

)

plt.show()

# ============================================================
# Actual vs Forecast Plot
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    train["ds"],

    train["y"],

    label="Training"

)

plt.plot(

    test["ds"],

    test["y"],

    label="Actual"

)

plt.plot(

    predictions["ds"],

    predictions["yhat"],

    label="Predicted"

)

plt.legend()

plt.grid()

plt.title(

    "Actual vs Predicted Sales"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Sales"

)

plt.show()

# ============================================================
# Next 30 Days Forecast
# ============================================================

future_30 = model.make_future_dataframe(

    periods=30,

    freq="D"

)

forecast_30 = model.predict(

    future_30

)

future_predictions = forecast_30.tail(

    30

)[

    [

        "ds",

        "yhat",

        "yhat_lower",

        "yhat_upper"

    ]

]

print("\n")

print("="*70)

print("Next 30 Days Forecast")

print("="*70)

display(

    future_predictions

)

# ============================================================
# Future Forecast Visualization
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    prophet_df["ds"],

    prophet_df["y"],

    label="Historical Sales"

)

plt.plot(

    future_predictions["ds"],

    future_predictions["yhat"],

    label="Future Forecast"

)

plt.fill_between(

    future_predictions["ds"],

    future_predictions["yhat_lower"],

    future_predictions["yhat_upper"],

    alpha=0.3

)

plt.legend()

plt.grid()

plt.title(

    "Next 30 Days Sales Forecast"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Sales"

)

plt.show()

# ============================================================
# Forecast Summary
# ============================================================

print("\n")

print("="*70)

print("Forecast Summary")

print("="*70)

display(

    future_predictions.describe()

)

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Prophet automatically detects trend and seasonality.")

print("- Weekly and yearly seasonal patterns improve forecasting accuracy.")

print("- Confidence intervals show uncertainty in future predictions.")

print("- Forecasts support inventory planning and sales strategy.")

print("- Retraining with new sales data improves long-term performance.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("Prophet Model Summary")

print("="*70)

print(

    "Algorithm               : Prophet"

)

print(

    "Training Records        :",

    len(train)

)

print(

    "Testing Records         :",

    len(test)

)

print(

    "Forecast Horizon        : 30 Days"

)

print(

    "MAE                     :",

    round(

        mae,

        2

    )

)

print(

    "MSE                     :",

    round(

        mse,

        2

    )

)

print(

    "RMSE                    :",

    round(

        rmse,

        2

    )

)

print("="*70)

# ============================================================
#  Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save Trained Prophet Model
# ============================================================

joblib.dump(

    model,

    "prophet_business_forecasting_model.joblib"

)

print("\n")

print("="*70)

print("Prophet Model Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Prophet Model
# ============================================================

loaded_model = joblib.load(

    "prophet_business_forecasting_model.joblib"

)

print("\n")

print("="*70)

print("Saved Prophet Model Loaded Successfully")

print("="*70)

# ============================================================
# Generate Next 30 Days
# ============================================================

future = loaded_model.make_future_dataframe(

    periods=30,

    freq="D"

)

# ============================================================
# Forecast Future Sales
# ============================================================

future_forecast = loaded_model.predict(

    future

)

forecast_output = future_forecast.tail(

    30

)[

    [

        "ds",

        "yhat",

        "yhat_lower",

        "yhat_upper"

    ]

]

print("\n")

print("="*70)

print("Next 30 Days Business Forecast")

print("="*70)

display(

    forecast_output

)

# ============================================================
# Export Forecast Results
# ============================================================

forecast_output.to_csv(

    "Future_Business_Forecast.csv",

    index=False

)

print("\n")

print("="*70)

print("Forecast Results Exported Successfully")

print("="*70)

# ============================================================
# Save Evaluation Metrics
# ============================================================

metrics = pd.DataFrame({

    "Metric":[

        "MAE",

        "MSE",

        "RMSE"

    ],

    "Value":[

        mae,

        mse,

        rmse

    ]

})

metrics.to_csv(

    "Evaluation_Metrics.csv",

    index=False

)

print("\n")

print("="*70)

print("Evaluation Metrics Saved Successfully")

print("="*70)

display(

    metrics

)

# ============================================================
# Forecast Visualization
# ============================================================

plt.figure(

    figsize=(15,6)

)

plt.plot(

    prophet_df["ds"],

    prophet_df["y"],

    label="Historical Sales"

)

plt.plot(

    forecast_output["ds"],

    forecast_output["yhat"],

    label="Forecast"

)

plt.fill_between(

    forecast_output["ds"],

    forecast_output["yhat_lower"],

    forecast_output["yhat_upper"],

    alpha=0.3

)

plt.legend()

plt.grid()

plt.title(

    "Historical vs Forecast Sales"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Sales"

)

plt.show()

# ============================================================
# Forecast Trend
# ============================================================

plt.figure(

    figsize=(12,5)

)

plt.plot(

    forecast_output["ds"],

    forecast_output["yhat"],

    marker="o"

)

plt.title(

    "30-Day Forecast Trend"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Forecast Sales"

)

plt.grid()

plt.show()

# ============================================================
# Forecast Statistics
# ============================================================

forecast_statistics = forecast_output.describe()

print("\n")

print("="*70)

print("Forecast Statistics")

print("="*70)

display(

    forecast_statistics

)

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Model",

        "Training Records",

        "Testing Records",

        "Forecast Horizon",

        "Prediction Interval"

    ],

    "Value":[

        "Time Series Forecasting",

        "Facebook Prophet",

        len(train),

        len(test),

        "30 Days",

        "95% Confidence"

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

        "Trained Prophet Model",

        "Business Forecast",

        "Evaluation Metrics"

    ],

    "Saved As":[

        "prophet_business_forecasting_model.joblib",

        "Future_Business_Forecast.csv",

        "Evaluation_Metrics.csv"

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

print("- Prophet automatically models trend, weekly, and yearly seasonality.")

print("- Forecast confidence intervals help quantify uncertainty.")

print("- Future sales forecasts support inventory planning and budgeting.")

print("- Businesses can optimize staffing and supply chain decisions.")

print("- Retraining the model periodically improves forecasting accuracy.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Business Forecasting using Prophet")

print("="*70)

print("Dataset                 : Superstore Sales Dataset")

print("Algorithm               : Facebook Prophet")

print("Training Records        :", len(train))

print("Testing Records         :", len(test))

print("Forecast Horizon        : 30 Days")

print("MAE                     :", round(mae,2))

print("MSE                     :", round(mse,2))

print("RMSE                    :", round(rmse,2))

print("Prediction Interval     : 95% Confidence")

print("Model Saved             : prophet_business_forecasting_model.joblib")

print("Forecast File           : Future_Business_Forecast.csv")

print("Metrics File            : Evaluation_Metrics.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Business Forecasting using Prophet Completed Successfully!")

print("="*70)



