# ============================================================
# Market Basket Analysis using Association Rule Mining (Apriori)
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

df = pd.read_csv(

    "Groceries_dataset.csv"

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

print("\n")

print("="*70)

print("Duplicate Records")

print("="*70)

duplicates = df.duplicated().sum()

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
# Unique Customers
# ============================================================

print("\n")

print("="*70)

print("Unique Customers")

print("="*70)

print(

    "Unique Customers :",

    df["Member_number"].nunique()

)

# ============================================================
# Unique Products
# ============================================================

print("\n")

print("="*70)

print("Unique Products")

print("="*70)

print(

    "Unique Products :",

    df["itemDescription"].nunique()

)

# ============================================================
# Top Selling Products
# ============================================================

top_products = df["itemDescription"].value_counts().head(20)

print("\n")

print("="*70)

print("Top 20 Selling Products")

print("="*70)

display(

    top_products

)

plt.figure(

    figsize=(12,6)

)

sns.barplot(

    x=top_products.values,

    y=top_products.index

)

plt.title(

    "Top 20 Selling Products"

)

plt.xlabel(

    "Purchase Count"

)

plt.ylabel(

    "Product"

)

plt.show()

# ============================================================
# Product Frequency Distribution
# ============================================================

print("\n")

print("="*70)

print("Product Frequency Distribution")

print("="*70)

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    df["itemDescription"].value_counts(),

    bins=30,

    kde=True

)

plt.title(

    "Product Frequency Distribution"

)

plt.xlabel(

    "Frequency"

)

plt.ylabel(

    "Number of Products"

)

plt.show()

# ============================================================
# Transactions Per Customer
# ============================================================

transactions = df.groupby(

    "Member_number"

).size()

print("\n")

print("="*70)

print("Transactions Per Customer")

print("="*70)

display(

    transactions.head()

)

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    transactions,

    bins=30,

    kde=True

)

plt.title(

    "Transactions Per Customer"

)

plt.xlabel(

    "Transactions"

)

plt.ylabel(

    "Customers"

)

plt.show()

# ============================================================
# Top 10 Active Customers
# ============================================================

top_customers = df["Member_number"].value_counts().head(10)

print("\n")

print("="*70)

print("Top 10 Active Customers")

print("="*70)

display(

    top_customers

)

plt.figure(

    figsize=(10,5)

)

sns.barplot(

    x=top_customers.index.astype(str),

    y=top_customers.values

)

plt.title(

    "Top 10 Active Customers"

)

plt.xlabel(

    "Customer ID"

)

plt.ylabel(

    "Purchase Count"

)

plt.xticks(

    rotation=45

)

plt.show()

# ============================================================
# Product Share (Top 10)
# ============================================================

top10 = df["itemDescription"].value_counts().head(10)

plt.figure(

    figsize=(8,8)

)

plt.pie(

    top10.values,

    labels=top10.index,

    autopct="%1.1f%%",

    startangle=90

)

plt.title(

    "Top 10 Product Share"

)

plt.show()

# ============================================================
# Monthly Purchase Trend
# ============================================================

df["Date"] = pd.to_datetime(

    df["Date"]

)

monthly_sales = df.groupby(

    df["Date"].dt.to_period("M")

).size()

print("\n")

print("="*70)

print("Monthly Purchase Trend")

print("="*70)

display(

    monthly_sales

)

plt.figure(

    figsize=(12,5)

)

monthly_sales.plot(

    marker="o"

)

plt.title(

    "Monthly Purchase Trend"

)

plt.xlabel(

    "Month"

)

plt.ylabel(

    "Transactions"

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

    "Total Records          :",

    df.shape[0]

)

print(

    "Total Features         :",

    df.shape[1]

)

print(

    "Unique Customers       :",

    df["Member_number"].nunique()

)

print(

    "Unique Products        :",

    df["itemDescription"].nunique()

)

print(

    "Most Sold Product      :",

    df["itemDescription"].mode()[0]

)

print(

    "Duplicate Records      :",

    duplicates

)

print("="*70)

# ============================================================
# Data Cleaning & Transaction Encoding
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from mlxtend.preprocessing import TransactionEncoder

# ============================================================
# Check Missing Values Before Treatment
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

df["itemDescription"].fillna(

    "Unknown",

    inplace=True

)

df["Member_number"].fillna(

    df["Member_number"].mode()[0],

    inplace=True

)

df["Date"].fillna(

    df["Date"].mode()[0],

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
# Remove Duplicate Records
# ============================================================

duplicates = df.duplicated().sum()

print("\n")

print("="*70)

print("Duplicate Records Before Removal")

print("="*70)

print("Duplicate Records :", duplicates)

df.drop_duplicates(

    inplace=True

)

print("\n")

print("="*70)

print("Duplicate Records Removed Successfully")

print("="*70)

print("Current Shape :", df.shape)

# ============================================================
# Create Transaction ID
# ============================================================

df["Transaction_ID"] = (

    df["Member_number"].astype(str)

    + "_"

    + df["Date"].astype(str)

)

print("\n")

print("="*70)

print("Transaction ID Created Successfully")

print("="*70)

display(

    df[

        [

            "Transaction_ID",

            "itemDescription"

        ]

    ].head()

)

# ============================================================
# Group Products by Transaction
# ============================================================

transactions = df.groupby(

    "Transaction_ID"

)[

    "itemDescription"

].apply(

    list

)

print("\n")

print("="*70)

print("Grouped Transactions")

print("="*70)

display(

    transactions.head()

)

# ============================================================
# Convert Transactions to List
# ============================================================

transaction_list = transactions.tolist()

print("\n")

print("="*70)

print("Transaction List Created")

print("="*70)

print(

    "Total Transactions :",

    len(transaction_list)

)

print("\n")

print("Sample Transaction")

print(transaction_list[0])

# ============================================================
# Transaction Encoder
# ============================================================

encoder = TransactionEncoder()

encoded_array = encoder.fit(

    transaction_list

).transform(

    transaction_list

)

print("\n")

print("="*70)

print("Transaction Encoding Completed")

print("="*70)

# ============================================================
# One-Hot Encoded DataFrame
# ============================================================

basket = pd.DataFrame(

    encoded_array,

    columns=encoder.columns_

)

print("\n")

print("="*70)

print("One-Hot Encoded Dataset")

print("="*70)

display(

    basket.head()

)

# ============================================================
# Dataset Shape
# ============================================================

print("\n")

print("="*70)

print("Encoded Dataset Shape")

print("="*70)

print("Rows    :", basket.shape[0])

print("Columns :", basket.shape[1])

# ============================================================
# Product Purchase Frequency
# ============================================================

product_frequency = basket.sum().sort_values(

    ascending=False

)

print("\n")

print("="*70)

print("Top 20 Frequently Purchased Products")

print("="*70)

display(

    product_frequency.head(20)

)

# ============================================================
# Top Products Visualization
# ============================================================

plt.figure(

    figsize=(12,6)

)

sns.barplot(

    x=product_frequency.head(20).values,

    y=product_frequency.head(20).index

)

plt.title(

    "Top 20 Purchased Products"

)

plt.xlabel(

    "Purchase Frequency"

)

plt.ylabel(

    "Products"

)

plt.show()

# ============================================================
# Transactions Size Analysis
# ============================================================

transaction_size = basket.sum(

    axis=1

)

print("\n")

print("="*70)

print("Transaction Size Statistics")

print("="*70)

display(

    transaction_size.describe()

)

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    transaction_size,

    bins=20,

    kde=True

)

plt.title(

    "Items Per Transaction"

)

plt.xlabel(

    "Number of Items"

)

plt.ylabel(

    "Transactions"

)

plt.show()

# ============================================================
# Sparsity Analysis
# ============================================================

total_cells = basket.shape[0] * basket.shape[1]

filled_cells = basket.values.sum()

sparsity = (

    1 -

    (filled_cells / total_cells)

) * 100

print("\n")

print("="*70)

print("Dataset Sparsity")

print("="*70)

print(

    "Sparsity :",

    round(

        sparsity,

        2

    ),

    "%"

)

# ============================================================
# Boolean Verification
# ============================================================

print("\n")

print("="*70)

print("Boolean Matrix Verification")

print("="*70)

display(

    basket.dtypes.head()

)

# ============================================================
# Dataset Summary
# ============================================================

print("\n")

print("="*70)

print("Dataset Summary")

print("="*70)

print(

    "Transactions             :",

    basket.shape[0]

)

print(

    "Products                 :",

    basket.shape[1]

)

print(

    "Most Purchased Product   :",

    product_frequency.idxmax()

)

print(

    "Highest Purchase Count   :",

    product_frequency.max()

)

print(

    "Average Basket Size      :",

    round(

        transaction_size.mean(),

        2

    )

)

print(

    "Dataset Sparsity (%)     :",

    round(

        sparsity,

        2

    )

)

print("\n")

print("Dataset Ready for Apriori Algorithm")

print("="*70)

# ============================================================
#  Apriori Model Building & Association Rule Mining
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

# ============================================================
# Generate Frequent Itemsets
# ============================================================

frequent_itemsets = apriori(

    basket,

    min_support=0.01,

    use_colnames=True

)

print("\n")

print("="*70)

print("Frequent Itemsets Generated Successfully")

print("="*70)

display(

    frequent_itemsets.head()

)

# ============================================================
# Frequent Itemsets Summary
# ============================================================

print("\n")

print("="*70)

print("Frequent Itemsets Summary")

print("="*70)

print(

    "Total Frequent Itemsets :",

    len(frequent_itemsets)

)

# ============================================================
# Top Frequent Itemsets
# ============================================================

top_itemsets = frequent_itemsets.sort_values(

    by="support",

    ascending=False

)

print("\n")

print("="*70)

print("Top 20 Frequent Itemsets")

print("="*70)

display(

    top_itemsets.head(20)

)

# ============================================================
# Support Visualization
# ============================================================

plt.figure(

    figsize=(12,6)

)

plt.bar(

    range(20),

    top_itemsets.head(20)["support"]

)

plt.xticks(

    range(20),

    [

        ", ".join(item)

        for item in top_itemsets.head(20)["itemsets"]

    ],

    rotation=90

)

plt.title(

    "Top 20 Frequent Itemsets"

)

plt.ylabel(

    "Support"

)

plt.show()

# ============================================================
# Generate Association Rules
# ============================================================

# Step 1: Load dataset
groceries = pd.read_csv("Groceries_dataset.csv")

# Step 2: Create basket (pivot table)
basket = pd.crosstab(groceries['Member_number'], groceries['itemDescription'])

# Step 3: Convert counts to boolean (saves memory vs int)
basket = (basket > 0)

# Step 4: Run fpgrowth (more memory-efficient than apriori)
# Adjust min_support to balance between too few and too many itemsets
frequent_itemsets = fpgrowth(basket, min_support=0.02, use_colnames=True, max_len=2)

print("\n")
print("="*70)
print("Frequent Itemsets Generated Successfully")
print("="*70)
print(frequent_itemsets.head())

# Step 5: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

print("\n")
print("="*70)
print("Association Rules Generated Successfully")
print("="*70)
print(rules.head())

# ============================================================
# Association Rules Summary
# ============================================================

print("\n")

print("="*70)

print("Association Rules Summary")

print("="*70)

print(

    "Total Rules :",

    len(rules)

)

# ============================================================
# Top Rules by Confidence
# ============================================================

top_confidence = rules.sort_values(

    by="confidence",

    ascending=False

)

print("\n")

print("="*70)

print("Top 20 Rules by Confidence")

print("="*70)

display(

    top_confidence.head(20)

)

# ============================================================
# Top Rules by Lift
# ============================================================

top_lift = rules.sort_values(

    by="lift",

    ascending=False

)

print("\n")

print("="*70)

print("Top 20 Rules by Lift")

print("="*70)

display(

    top_lift.head(20)

)

# ============================================================
# Support vs Confidence
# ============================================================

plt.figure(

    figsize=(10,6)

)

sns.scatterplot(

    data=rules,

    x="support",

    y="confidence",

    size="lift",

    hue="lift",

    sizes=(20,300)

)

plt.title(

    "Support vs Confidence"

)

plt.show()

# ============================================================
# Lift Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    rules["lift"],

    bins=30,

    kde=True

)

plt.title(

    "Lift Distribution"

)

plt.xlabel(

    "Lift"

)

plt.ylabel(

    "Frequency"

)

plt.show()

# ============================================================
# Leverage Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    rules["leverage"],

    bins=30,

    kde=True

)

plt.title(

    "Leverage Distribution"

)

plt.show()

# ============================================================
# Conviction Distribution
# ============================================================

plt.figure(

    figsize=(10,5)

)

sns.histplot(

    rules["conviction"],

    bins=30,

    kde=True

)

plt.title(

    "Conviction Distribution"

)

plt.show()

# ============================================================
# Recommendation Examples
# ============================================================

recommendations = rules[

    [

        "antecedents",

        "consequents",

        "support",

        "confidence",

        "lift"

    ]

]

print("\n")

print("="*70)

print("Sample Product Recommendations")

print("="*70)

display(

    recommendations.head(20)

)

# ============================================================
# Top Strong Rules
# ============================================================

strong_rules = rules[

    rules["lift"] > 1

].sort_values(

    by="lift",

    ascending=False

)

print("\n")

print("="*70)

print("Top Strong Association Rules")

print("="*70)

display(

    strong_rules.head(20)

)

# ============================================================
# Rule Length Analysis
# ============================================================

rules["Rule_Length"] = (

    rules["antecedents"].apply(len)

    +

    rules["consequents"].apply(len)

)

plt.figure(

    figsize=(8,5)

)

sns.countplot(

    x="Rule_Length",

    data=rules

)

plt.title(

    "Rule Length Distribution"

)

plt.show()

# ============================================================
# Business Insights
# ============================================================

print("\n")

print("="*70)

print("Business Insights")

print("="*70)

print("- Frequently purchased products can be placed near each other.")

print("- High lift indicates strong product associations.")

print("- High confidence rules improve cross-selling opportunities.")

print("- Strong rules help build recommendation engines.")

print("- Product bundles can increase average basket value.")

# ============================================================
# Final Summary
# ============================================================

print("\n")

print("="*70)

print("Apriori Algorithm Summary")

print("="*70)

print("Transactions            :", basket.shape[0])

print("Products                :", basket.shape[1])

print("Frequent Itemsets       :", len(frequent_itemsets))

print("Association Rules       :", len(rules))

print("Minimum Support         : 0.01")

print("Minimum Confidence      : 0.30")

print("="*70)

# ============================================================
# ============================================================
# Market Basket Analysis using Association Rule Mining (Apriori)
# Part-3 : Deployment
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import joblib

# ============================================================
# Save Frequent Itemsets
# ============================================================

joblib.dump(

    frequent_itemsets,

    "frequent_itemsets.joblib"

)

# ============================================================
# Save Association Rules
# ============================================================

joblib.dump(

    rules,

    "association_rules.joblib"

)

print("\n")

print("="*70)

print("Frequent Itemsets and Association Rules Saved Successfully")

print("="*70)

# ============================================================
# Load Saved Files
# ============================================================

loaded_itemsets = joblib.load(

    "frequent_itemsets.joblib"

)

loaded_rules = joblib.load(

    "association_rules.joblib"

)

print("\n")

print("="*70)

print("Saved Files Loaded Successfully")

print("="*70)

# ============================================================
# Product Recommendation Function
# ============================================================

def recommend_products(product_name):

    recommendations = loaded_rules[

        loaded_rules["antecedents"].apply(

            lambda x: product_name in x

        )

    ]

    recommendations = recommendations.sort_values(

        by="lift",

        ascending=False

    )

    if recommendations.empty:

        print("\nNo recommendations found.")

        return

    result = recommendations[

        [

            "antecedents",

            "consequents",

            "support",

            "confidence",

            "lift"

        ]

    ]

    return result

# ============================================================
# Test Recommendation
# ============================================================

sample_product = basket.columns[0]

print("\n")

print("="*70)

print("Sample Product Recommendation")

print("="*70)

print("Selected Product :", sample_product)

recommendation_output = recommend_products(

    sample_product

)

display(

    recommendation_output

)

# ============================================================
# Export Frequent Itemsets
# ============================================================

frequent_itemsets.to_csv(

    "Frequent_Itemsets.csv",

    index=False

)

print("\n")

print("="*70)

print("Frequent Itemsets Exported Successfully")

print("="*70)

# ============================================================
# Export Association Rules
# ============================================================

rules.to_csv(

    "Association_Rules.csv",

    index=False

)

print("\n")

print("="*70)

print("Association Rules Exported Successfully")

print("="*70)

# ============================================================
# Top 20 Rules
# ============================================================

top_rules = rules.sort_values(

    by="lift",

    ascending=False

).head(20)

print("\n")

print("="*70)

print("Top 20 Association Rules")

print("="*70)

display(

    top_rules

)

top_rules.to_csv(

    "Top_20_Association_Rules.csv",

    index=False

)

# ============================================================
# Model Information
# ============================================================

model_information = pd.DataFrame({

    "Parameter":[

        "Algorithm",

        "Minimum Support",

        "Minimum Confidence",

        "Transactions",

        "Products"

    ],

    "Value":[

        "Apriori",

        0.01,

        0.30,

        basket.shape[0],

        basket.shape[1]

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

        "Frequent Itemsets",

        "Association Rules",

        "Frequent Itemsets CSV",

        "Association Rules CSV",

        "Top 20 Rules"

    ],

    "Saved As":[

        "frequent_itemsets.joblib",

        "association_rules.joblib",

        "Frequent_Itemsets.csv",

        "Association_Rules.csv",

        "Top_20_Association_Rules.csv"

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

print("- Frequently purchased products should be placed close together.")

print("- High lift rules indicate strong product relationships.")

print("- High confidence rules improve recommendation quality.")

print("- Retailers can create combo offers using strong associations.")

print("- Association rules help increase cross-selling and average basket value.")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)

print("Market Basket Analysis using Apriori")

print("="*70)

print("Dataset                  : Groceries Dataset")

print("Algorithm                : Apriori")

print("Transactions             :", basket.shape[0])

print("Products                 :", basket.shape[1])

print("Frequent Itemsets        :", len(frequent_itemsets))

print("Association Rules        :", len(rules))

print("Minimum Support          : 0.01")

print("Minimum Confidence       : 0.30")

print("Top Rule Lift            :", round(rules['lift'].max(),4))

print("Files Saved              : frequent_itemsets.joblib")

print("Rules Saved              : association_rules.joblib")

print("Project Status           : Deployment Ready")

print("="*70)

# ============================================================
# Project Completed
# ============================================================

print("\n")

print("="*70)

print("Market Basket Analysis using Apriori Completed Successfully!")

print("="*70)
