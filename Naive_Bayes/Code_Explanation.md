# ============================================================
# SMS Spam Detection using Multinomial Naive Bayes
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inside Jupyter Notebook
%matplotlib inline

# Set plot style
sns.set_style("whitegrid")

# ============================================================
# Load Dataset
# ============================================================

# The dataset is tab-separated and does not contain column names.

df = pd.read_csv(

    "SMSSpamCollection",

    sep="\t",

    header=None,

    names=["label", "message"]

)

# ============================================================
# Basic Dataset Exploration
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
# Duplicate Messages
# ============================================================

print("\n")

print("="*60)
print("Duplicate Messages")
print("="*60)

print(df.duplicated().sum())

# Remove duplicate messages

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")

print(df.shape)

# ============================================================
# Target Distribution
# ============================================================

print("\n")

print("="*60)
print("Spam vs Ham")
print("="*60)

print(df["label"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(

    data=df,

    x="label"

)

plt.title("Spam vs Ham Messages")

plt.xlabel("Message Type")

plt.ylabel("Count")

plt.show()

# ============================================================
# Convert Labels into Numbers
# ============================================================

# ham = 0
# spam = 1

df["label"] = df["label"].map({

    "ham":0,

    "spam":1

})

print("\n")

print("="*60)
print("Encoded Labels")
print("="*60)

print(df.head())

# ============================================================
# Character Count
# ============================================================

df["Character_Count"] = df["message"].apply(len)

# ============================================================
# Word Count
# ============================================================

df["Word_Count"] = df["message"].apply(

    lambda x: len(x.split())

)

# ============================================================
# Sentence Count
# ============================================================

df["Sentence_Count"] = df["message"].apply(

    lambda x: x.count(".") +

              x.count("!") +

              x.count("?") + 1

)

# ============================================================
# View New Features
# ============================================================

print("\n")

print("="*60)
print("Dataset with New Features")
print("="*60)

print(df.head())

# ============================================================
# Statistical Summary
# ============================================================

print("\n")

print("="*60)
print("Statistical Summary")
print("="*60)

print(

    df[[

        "Character_Count",

        "Word_Count",

        "Sentence_Count"

    ]].describe()

)

# ============================================================
# Compare Ham vs Spam
# ============================================================

print("\n")

print("="*60)
print("Grouped Statistics")
print("="*60)

print(

    df.groupby(

        "label"

    )[

        [

            "Character_Count",

            "Word_Count",

            "Sentence_Count"

        ]

    ].mean()

)

# ============================================================
# Character Count Distribution
# ============================================================

plt.figure(figsize=(7,5))

sns.histplot(

    data=df,

    x="Character_Count",

    hue="label",

    bins=50,

    kde=True

)

plt.title("Character Count Distribution")

plt.show()

# ============================================================
# Word Count Distribution
# ============================================================

plt.figure(figsize=(7,5))

sns.histplot(

    data=df,

    x="Word_Count",

    hue="label",

    bins=40,

    kde=True

)

plt.title("Word Count Distribution")

plt.show()

# ============================================================
# Pair Plot
# ============================================================

sns.pairplot(

    df[[

        "Character_Count",

        "Word_Count",

        "Sentence_Count",

        "label"

    ]],

    hue="label"

)

plt.show()

# ============================================================
# Correlation
# ============================================================

plt.figure(figsize=(6,5))

sns.heatmap(

    df[[

        "Character_Count",

        "Word_Count",

        "Sentence_Count",

        "label"

    ]].corr(),

    annot=True,

    cmap="coolwarm"

)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# Prepare Features and Target
# ============================================================

# Input Feature

X = df["message"]

# Target Variable

y = df["label"]

print("\n")

print("="*60)
print("Sample Messages")
print("="*60)

print(X.head())

print("\n")

print("="*60)
print("Target Labels")
print("="*60)

print(y.head())

print("\n")

print("Input Shape :", X.shape)

print("Target Shape:", y.shape)


# ============================================================
# Install NLTK (Run only once)
# ============================================================

# pip install nltk

# ============================================================
# Import Required Libraries
# ============================================================

import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (Run only once)

nltk.download("punkt")
nltk.download("stopwords")

# Create Stemmer

stemmer = PorterStemmer()

# ============================================================
# Text Preprocessing Function
# ============================================================

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    words = [

        word

        for word in words

        if word not in stopwords.words("english")

    ]

    # Apply stemming
    words = [

        stemmer.stem(word)

        for word in words

    ]

    # Join words back into a sentence
    return " ".join(words)

# ============================================================
# Apply Preprocessing
# ============================================================

df["processed_message"] = df["message"].apply(

    preprocess_text

)

print("="*70)
print("Original vs Processed Messages")
print("="*70)

comparison = pd.DataFrame({

    "Original": df["message"].head(),

    "Processed": df["processed_message"].head()

})

print(comparison)

# ============================================================
# TF-IDF Vectorization
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(

    df["processed_message"]

)

y = df["label"]

print("\n")

print("="*70)
print("TF-IDF Matrix Shape")
print("="*70)

print(X.shape)

# ============================================================
# Train-Test Split
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\n")

print("="*70)
print("Training Shape")
print("="*70)

print(X_train.shape)

print(y_train.shape)

print("\n")

print("="*70)
print("Testing Shape")
print("="*70)

print(X_test.shape)

print(y_test.shape)

# ============================================================
# Train Multinomial Naive Bayes
# ============================================================

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(

    X_train,

    y_train

)

print("\nModel Training Completed Successfully!")

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(

    X_test

)

# Probability Prediction

y_prob = model.predict_proba(

    X_test

)

print("\n")

print("="*70)
print("Predicted Labels")
print("="*70)

print(y_pred[:10])

print("\n")

print("="*70)
print("Prediction Probabilities")
print("="*70)

print(y_prob[:10])

# ============================================================
# Model Evaluation
# ============================================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score

)

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
print("Model Performance")
print("="*70)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

# ============================================================
# Cross Validation
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([

    ("tfidf", TfidfVectorizer()),

    ("model", MultinomialNB())

])

cv_scores = cross_val_score(

    pipeline,

    df["processed_message"],

    y,

    cv=5,

    scoring="accuracy"

)

print("\n")

print("="*70)
print("Cross Validation")
print("="*70)

print(cv_scores)

print("\nAverage Accuracy")

print(cv_scores.mean())

# ============================================================
# Top Important Words
# ============================================================

feature_names = tfidf.get_feature_names_out()

spam_scores = model.feature_log_prob_[1]

top_words = pd.DataFrame({

    "Word": feature_names,

    "Importance": spam_scores

})

top_words = top_words.sort_values(

    by="Importance",

    ascending=False

)

print("\n")

print("="*70)
print("Top 20 Spam Words")
print("="*70)

print(top_words.head(20))


# ============================================================
# Confusion Matrix
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(

    y_test,

    y_pred

)

print("="*70)
print("Confusion Matrix")
print("="*70)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=["Ham","Spam"],

    yticklabels=["Ham","Spam"]

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# Classification Report
# ============================================================

from sklearn.metrics import classification_report

print("\n")

print("="*70)
print("Classification Report")
print("="*70)

print(

    classification_report(

        y_test,

        y_pred,

        target_names=["Ham","Spam"]

    )

)

# ============================================================
# ROC Curve
# ============================================================

from sklearn.metrics import (

    roc_curve,

    roc_auc_score

)

# Probability of Spam class

spam_probability = y_prob[:,1]

fpr, tpr, thresholds = roc_curve(

    y_test,

    spam_probability

)

auc_score = roc_auc_score(

    y_test,

    spam_probability

)

plt.figure(figsize=(7,6))

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
print("ROC AUC Score")
print("="*70)

print(auc_score)

# ============================================================
# Compare Actual vs Predicted
# ============================================================

comparison = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": y_pred

})

print("\n")

print("="*70)
print("Sample Predictions")
print("="*70)

print(comparison.head(20))

# ============================================================
# Predict New SMS
# ============================================================

new_sms = [

    "Congratulations! You have won a FREE iPhone. Call now to claim your prize."

]

# Apply the same preprocessing

processed_sms = [

    preprocess_text(message)

    for message in new_sms

]

# Convert text into TF-IDF features

sms_vector = tfidf.transform(

    processed_sms

)

# Predict

prediction = model.predict(

    sms_vector

)

# Prediction Probability

prediction_probability = model.predict_proba(

    sms_vector

)

print("\n")

print("="*70)
print("Prediction for New SMS")
print("="*70)

print("Message")

print(new_sms[0])

print("\nPredicted Class :", prediction[0])

if prediction[0] == 0:

    print("Result : HAM (Legitimate Message)")

else:

    print("Result : SPAM Message")

print("\nPrediction Probability")

print(prediction_probability)

# ============================================================
# Save Model
# ============================================================

from joblib import dump

dump(

    model,

    "spam_detection_model.joblib"

)

dump(

    tfidf,

    "tfidf_vectorizer.joblib"

)

print("\nModel Saved Successfully!")

# ============================================================
# Load Model
# ============================================================

from joblib import load

loaded_model = load(

    "spam_detection_model.joblib"

)

loaded_vectorizer = load(

    "tfidf_vectorizer.joblib"

)

print("Saved Model Loaded Successfully!")

# ============================================================
# Predict Using Loaded Model
# ============================================================

loaded_vector = loaded_vectorizer.transform(

    processed_sms

)

loaded_prediction = loaded_model.predict(

    loaded_vector

)

print("\nPrediction Using Loaded Model")

print(loaded_prediction)

# ============================================================
# Test Multiple Messages
# ============================================================

sample_messages = [

    "Hey, are you free this evening?",

    "URGENT! Claim your FREE reward now.",

    "Meeting has been postponed to tomorrow.",

    "You have won $1000 cash. Click the link now!"

]

print("\n")

print("="*70)
print("Multiple SMS Predictions")
print("="*70)

for sms in sample_messages:

    processed = preprocess_text(sms)

    vector = loaded_vectorizer.transform([processed])

    result = loaded_model.predict(vector)[0]

    probability = loaded_model.predict_proba(vector)[0]

    print(f"\nMessage : {sms}")

    if result == 0:

        print("Prediction : HAM")

    else:

        print("Prediction : SPAM")

    print(f"Probability : {probability}")

# ============================================================
# Final Project Summary
# ============================================================

print("\n")

print("="*70)
print("SMS Spam Detection using Multinomial Naive Bayes")
print("="*70)

print(f"Total Messages           : {len(df)}")

print(f"Ham Messages             : {(df['label']==0).sum()}")

print(f"Spam Messages            : {(df['label']==1).sum()}")

print(f"\nTraining Samples        : {X_train.shape[0]}")

print(f"Testing Samples         : {X_test.shape[0]}")

print(f"TF-IDF Features         : {X_train.shape[1]}")

print(f"\nAccuracy               : {accuracy:.4f}")

print(f"Precision              : {precision:.4f}")

print(f"Recall                 : {recall:.4f}")

print(f"F1 Score               : {f1:.4f}")

print(f"ROC AUC Score          : {auc_score:.4f}")

print(f"Cross Validation       : {cv_scores.mean():.4f}")

print("\nModel Status : Ready for Deployment")

print("="*70)

print("\nProject Completed Successfully!")
