# ============================================================
# Stock Price Forecasting using ARIMA
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

    "RELIANCE.csv"

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
# Convert Date Column
# ============================================================

df["Date"] = pd.to_datetime(

    df["Date"]

)

print("\n")

print("="*70)

print("Date Converted Successfully")

print("="*70)

# ============================================================
# Sort Dataset by Date
# ============================================================

df = df.sort_values(

    by="Date"

)

print("\n")

print("="*70)

print("Dataset Sorted by Date")

print("="*70)

# ============================================================
# Closing Price Trend
# ============================================================

plt.figure(

    figsize=(14,6)

)

plt.plot(

    df["Date"],

    df["Close"]

)

plt.title(

    "Closing Price Trend"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Close Price"

)

plt.grid()

plt.show()

# ============================================================
# Daily Returns
# ============================================================

df["Daily_Return"] = df["Close"].pct_change()

print("\n")

print("="*70)

print("Daily Returns")

print("="*70)

display(

    df[

        [

            "Date",

            "Close",

            "Daily_Return"

        ]

    ].head()

)

# ============================================================
# Daily Return Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    df["Daily_Return"].dropna(),

    bins=40,

    kde=True

)

plt.title(

    "Daily Return Distribution"

)

plt.xlabel(

    "Daily Return"

)

plt.show()

# ============================================================
# Closing Price Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    df["Close"],

    bins=40,

    kde=True

)

plt.title(

    "Closing Price Distribution"

)

plt.xlabel(

    "Closing Price"

)

plt.show()

# ============================================================
# Volume Analysis
# ============================================================

plt.figure(

    figsize=(14,5)

)

plt.plot(

    df["Date"],

    df["Volume"]

)

plt.title(

    "Trading Volume"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Volume"

)

plt.grid()

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
# Open vs Close Price
# ============================================================

plt.figure(

    figsize=(8,6)

)

sns.scatterplot(

    data=df,

    x="Open",

    y="Close"

)

plt.title(

    "Open Price vs Close Price"

)

plt.show()

# ============================================================
# Highest Closing Price
# ============================================================

highest_close = df.loc[

    df["Close"].idxmax()

]

print("\n")

print("="*70)

print("Highest Closing Price")

print("="*70)

display(

    highest_close

)

# ============================================================
# Lowest Closing Price
# ============================================================

lowest_close = df.loc[

    df["Close"].idxmin()

]

print("\n")

print("="*70)

print("Lowest Closing Price")

print("="*70)

display(

    lowest_close

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Total Records          :",

    df.shape[0]

)

print(

    "Total Features         :",

    df.shape[1]

)

print(

    "Date Range             :",

    df["Date"].min(),

    "to",

    df["Date"].max()

)

print(

    "Highest Closing Price  :",

    round(

        df["Close"].max(),

        2

    )

)

print(

    "Lowest Closing Price   :",

    round(

        df["Close"].min(),

        2

    )

)

print(

    "Average Closing Price  :",

    round(

        df["Close"].mean(),

        2

    )

)

print(

    "Average Daily Volume   :",

    round(

        df["Volume"].mean(),

        2

    )

)

print("="*70)

# ============================================================
# Time Series Preprocessing & Stationarity Analysis
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

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

# ============================================================
# Set Date as Index
# ============================================================

df.set_index(

    "Date",

    inplace=True

)

print("\n")

print("="*70)

print("Date Set as Index")

print("="*70)

display(

    df.head()

)

# ============================================================
# Select Closing Price
# ============================================================

ts = df["Close"]

print("\n")

print("="*70)

print("Time Series Created")

print("="*70)

display(

    ts.head()

)

# ============================================================
# Time Series Visualization
# ============================================================

plt.figure(

    figsize=(14,6)

)

plt.plot(

    ts

)

plt.title(

    "Closing Price Time Series"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Closing Price"

)

plt.grid()

plt.show()

# ============================================================
# Rolling Mean
# ============================================================

rolling_mean = ts.rolling(

    window=30

).mean()

rolling_std = ts.rolling(

    window=30

).std()

plt.figure(

    figsize=(14,6)

)

plt.plot(

    ts,

    label="Original"

)

plt.plot(

    rolling_mean,

    label="Rolling Mean"

)

plt.plot(

    rolling_std,

    label="Rolling Std"

)

plt.legend()

plt.title(

    "Rolling Mean and Standard Deviation"

)

plt.show()

# ============================================================
# Augmented Dickey-Fuller Test
# ============================================================

print("\n")

print("="*70)

print("ADF Test")

print("="*70)

adf_result = adfuller(

    ts

)

print(

    "ADF Statistic :", adf_result[0]

)

print(

    "p-value :", adf_result[1]

)

print(

    "Critical Values"

)

for key, value in adf_result[4].items():

    print(

        key,

        ":",

        value

    )

if adf_result[1] < 0.05:

    print("\nTime Series is Stationary")

else:

    print("\nTime Series is NOT Stationary")

# ============================================================
# Differencing
# ============================================================

ts_diff = ts.diff().dropna()

print("\n")

print("="*70)

print("Differencing Applied")

print("="*70)

display(

    ts_diff.head()

)

# ============================================================
# Differenced Series
# ============================================================

plt.figure(

    figsize=(14,6)

)

plt.plot(

    ts_diff

)

plt.title(

    "Differenced Time Series"

)

plt.grid()

plt.show()

# ============================================================
# ADF Test After Differencing
# ============================================================

print("\n")

print("="*70)

print("ADF Test After Differencing")

print("="*70)

adf_diff = adfuller(

    ts_diff

)

print(

    "ADF Statistic :",

    adf_diff[0]

)

print(

    "p-value :",

    adf_diff[1]

)

if adf_diff[1] < 0.05:

    print("\nDifferenced Series is Stationary")

else:

    print("\nDifferenced Series is NOT Stationary")

# ============================================================
# ACF Plot
# ============================================================

plt.figure(

    figsize=(10,5)

)

plot_acf(

    ts_diff,

    lags=40

)

plt.show()

# ============================================================
# PACF Plot
# ============================================================

plt.figure(

    figsize=(10,5)

)

plot_pacf(

    ts_diff,

    lags=40,

    method="ywm"

)

plt.show()

# ============================================================
# Train-Test Split
# ============================================================

train_size = int(

    len(ts) * 0.80

)

train = ts.iloc[:train_size]

test = ts.iloc[train_size:]

print("\n")

print("="*70)

print("Train-Test Split")

print("="*70)

print(

    "Training Samples :",

    len(train)

)

print(

    "Testing Samples :",

    len(test)

)

# ============================================================
# Train vs Test Visualization
# ============================================================

plt.figure(

    figsize=(14,6)

)

plt.plot(

    train,

    label="Train"

)

plt.plot(

    test,

    label="Test"

)

plt.legend()

plt.title(

    "Train and Test Data"

)

plt.grid()

plt.show()

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Total Observations :", len(ts)

)

print(

    "Training Samples   :", len(train)

)

print(

    "Testing Samples    :", len(test)

)

print(

    "ADF p-value        :", round(adf_result[1],6)

)

print(

    "Differenced p-value:", round(adf_diff[1],6)

)

print("\n")

print("Dataset Ready for ARIMA Model")

print("="*70)

# ============================================================
# ARIMA Model Building & Forecasting
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import numpy as np

# ============================================================
# Build ARIMA Model
# ============================================================

model = ARIMA(

    train,

    order=(5,1,0)

)

print("\n")

print("="*70)

print("ARIMA Model Created Successfully")

print("="*70)

# ============================================================
# Train Model
# ============================================================

model_fit = model.fit()

print("\n")

print("="*70)

print("Model Training Completed")

print("="*70)

# ============================================================
# Model Summary
# ============================================================

print("\n")

print("="*70)

print("ARIMA Model Summary")

print("="*70)

print(

    model_fit.summary()

)

# ============================================================
# Forecast Test Data
# ============================================================

forecast = model_fit.forecast(

    steps=len(test)

)

forecast = pd.Series(

    forecast,

    index=test.index

)

print("\n")

print("="*70)

print("Forecast Generated Successfully")

print("="*70)

display(

    forecast.head()

)

# ============================================================
# Evaluation Metrics
# ============================================================
test = np.nan_to_num(test, nan=0.0)
forecast = np.nan_to_num(forecast, nan=0.0)

mae = mean_absolute_error(

    test,

    forecast

)

mse = mean_squared_error(

    test,

    forecast

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

        4

    )

)

print(

    "MSE  :",

    round(

        mse,

        4

    )

)

print(

    "RMSE :",

    round(

        rmse,

        4

    )

)

# ============================================================
# Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual":

    test,

    "Predicted":

    forecast

})

print("\n")

print("="*70)

print("Actual vs Predicted")

print("="*70)

display(

    comparison.head(20)

)

# ============================================================
# Actual vs Forecast Plot
# ============================================================

plt.figure(

    figsize=(14,6)

)

plt.plot(

    train,

    label="Train"

)

plt.plot(

    test,

    label="Actual"

)

plt.plot(

    forecast,

    label="Forecast"

)

plt.legend()

plt.title(

    "Actual vs Forecast"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Closing Price"

)

plt.grid()

plt.show()

# ============================================================
# Forecast Next 30 Days
# ============================================================

# Assume your original time series is called 'df' with a Date index
last_date = df.index[-1]   # get last date from the full dataset

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_forecast = model_fit.forecast(steps=30)

future_predictions = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_forecast
})

print("\n")
print("="*70)
print("Next 30 Days Forecast")
print("="*70)
display(future_predictions)


# ============================================================
# Future Forecast Visualization
# ============================================================

plt.figure(

    figsize=(14,6)

)

plt.plot(

    ts,

    label="Historical"

)

plt.plot(

    future_dates,

    future_forecast,

    label="Future Forecast"

)

plt.legend()

plt.title(

    "Next 30 Days Stock Forecast"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Closing Price"

)

plt.grid()

plt.show()

# ============================================================
# Residual Analysis
# ============================================================

residuals = model_fit.resid

print("\n")

print("="*70)

print("Residual Statistics")

print("="*70)

display(

    residuals.describe()

)

# ============================================================
# Residual Plot
# ============================================================

plt.figure(

    figsize=(12,5)

)

plt.plot(

    residuals

)

plt.title(

    "Residual Errors"

)

plt.grid()

plt.show()

# ============================================================
# Residual Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    residuals,

    bins=40,

    kde=True

)

plt.title(

    "Residual Distribution"

)

plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- ARIMA captures historical price trends for forecasting.")

print("- Lower MAE and RMSE indicate better forecasting accuracy.")

print("- Residuals should resemble random noise for a well-fitted model.")

print("- Forecasts assist investors in short-term decision making.")

print("- ARIMA performs best on stationary time series data.")

# ============================================================
# Final Model Summary
# ============================================================

print("\n")

print("="*70)

print("ARIMA Model Summary")

print("="*70)

print(

    "Algorithm              : ARIMA"

)

print(

    "Order (p,d,q)          : (5,1,0)"

)

print(

    "Training Samples       :",

    len(train)

)

print(

    "Testing Samples        :",

    len(test)

)

print(

    "MAE                    :",

    round(

        mae,

        4

    )

)

print(

    "MSE                    :",

    round(

        mse,

        4

    )

)

print(

    "RMSE                   :",

    round(

        rmse,

        4

    )

)

print(

    "Forecast Horizon       : 30 Days"

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
# Save Trained ARIMA Model
# ============================================================

joblib.dump(

    model_fit,

    "arima_stock_model.joblib"

)

print("\n")

print("="*70)

print("ARIMA Model Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Model
# ============================================================

loaded_model = joblib.load(

    "arima_stock_model.joblib"

)

print("\n")

print("="*70)

print("Saved ARIMA Model Loaded Successfully")

print("="*70)

# ============================================================
# Forecast Next 30 Days
# ============================================================

future_forecast = loaded_model.forecast(

    steps=30

)

future_dates = pd.date_range(

    start=ts.index[-1] + pd.Timedelta(days=1),

    periods=30,

    freq="D"

)

forecast_df = pd.DataFrame({

    "Date": future_dates,

    "Forecast_Close_Price": future_forecast

})

print("\n")

print("="*70)

print("Next 30 Days Forecast")

print("="*70)

display(

    forecast_df

)

# ============================================================
# Export Forecast Results
# ============================================================

forecast_df.to_csv(

    "Future_Stock_Forecast.csv",

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

    figsize=(14,6)

)

plt.plot(

    ts,

    label="Historical Price"

)

plt.plot(

    future_dates,

    future_forecast,

    label="30-Day Forecast"

)

plt.title(

    "Historical vs Forecasted Stock Prices"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Closing Price"

)

plt.legend()

plt.grid()

plt.show()

# ============================================================
# Forecast Trend
# ============================================================

plt.figure(

    figsize=(10,5)

)

plt.plot(

    forecast_df["Date"],

    forecast_df["Forecast_Close_Price"],

    marker="o"

)

plt.title(

    "Forecasted Closing Prices"

)

plt.xlabel(

    "Date"

)

plt.ylabel(

    "Forecast Price"

)

plt.grid()

plt.show()

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Model",

        "Order (p,d,q)",

        "Training Samples",

        "Testing Samples",

        "Forecast Horizon"

    ],

    "Value":[

        "Time Series Forecasting",

        "ARIMA",

        "(5,1,0)",

        len(train),

        len(test),

        "30 Days"

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
# Deployment Files
# ============================================================

deployment_summary = pd.DataFrame({

    "File":[

        "Trained Model",

        "Future Forecast",

        "Evaluation Metrics"

    ],

    "Saved As":[

        "arima_stock_model.joblib",

        "Future_Stock_Forecast.csv",

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

print("- ARIMA captures historical price movements for short-term forecasting.")

print("- Lower MAE, MSE, and RMSE indicate better predictive performance.")

print("- Forecasts can assist analysts in planning and trend analysis.")

print("- Time series models should be retrained regularly with new market data.")

print("- Forecasts should support decisions, not replace risk management.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Stock Price Forecasting using ARIMA")

print("="*70)

print("Dataset                 : NIFTY 50 Stock Data")

print("Algorithm               : ARIMA")

print("Order (p,d,q)           : (5,1,0)")

print("Training Samples        :", len(train))

print("Testing Samples         :", len(test))

print("Forecast Horizon        : 30 Days")

print("MAE                     :", round(mae,4))

print("MSE                     :", round(mse,4))

print("RMSE                    :", round(rmse,4))

print("Model Saved             : arima_stock_model.joblib")

print("Forecast File           : Future_Stock_Forecast.csv")

print("Metrics File            : Evaluation_Metrics.csv")

print("Project Status          : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Stock Price Forecasting using ARIMA Completed Successfully!")

print("="*70)


