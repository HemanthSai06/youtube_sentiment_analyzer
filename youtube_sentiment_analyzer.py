
# ============================================================
# YOUTUBE COMMENTS SENTIMENT ANALYZER USING MACHINE LEARNING
#
# Algorithms:
# 1. Multinomial Naive Bayes
# 2. Logistic Regression
# 3. Support Vector Machine (Linear SVM)
#
# Feature Extraction: TF-IDF
# Learning Type: Supervised Learning
# Problem Type: Multiclass Classification
#
# IDE: Spyder
#
# IMPORTANT:
# No files, plots, graphs or results are saved.
# Everything is displayed inside Spyder.
# ============================================================


# ==================== 1. IMPORT LIBRARIES ====================

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tkinter import Tk, filedialog


# ==================== 2. LOAD DATASET ====================

print("\n")
print("=" * 65)
print("YOUTUBE COMMENTS SENTIMENT ANALYZER")
print("=" * 65)

root = Tk()
root.withdraw()

file_path = ("/Users/phemanthsai/Desktop/Ml_Pro/youtube_sentiment_analyzer/youtube_comments_cleaned.csv"
)

root.destroy()

if not file_path:
    raise SystemExit("No dataset selected. Program terminated.")

df = pd.read_csv(file_path)

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:", df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst Five Records:")
print(df.head())


# ==================== 3. IDENTIFY COLUMNS ====================

def find_column(dataframe, possible_names):

    column_map = {
        str(col).strip().lower().replace("_", "").replace(" ", ""): col
        for col in dataframe.columns
    }

    for name in possible_names:

        key = name.lower().replace("_", "").replace(" ", "")

        if key in column_map:
            return column_map[key]

    return None


text_col = find_column(
    df,
    ["CommentText", "Comment", "Text",
     "Comments", "Comment_Content"]
)

target_col = find_column(
    df,
    ["Sentiment", "Label", "Target", "Class"]
)


if text_col is None or target_col is None:

    raise ValueError(
        "Required columns not found!\n"
        "Available columns: " + str(df.columns.tolist())
    )


print("\nInput Feature:", text_col)
print("Target Variable:", target_col)


# ==================== 4. DATA CLEANING ====================

print("\n")
print("=" * 65)
print("DATA PREPROCESSING")
print("=" * 65)

# Select only required columns
df = df[[text_col, target_col]].copy()

# Remove missing values
df.dropna(subset=[text_col, target_col], inplace=True)

# Convert columns into strings
df[text_col] = df[text_col].astype(str)

df[target_col] = (
    df[target_col].astype(str).str.strip().str.lower()
)

# Remove empty comments
df = df[df[text_col].str.strip() != ""]

# Remove duplicate comment-label pairs
df.drop_duplicates(
    subset=[text_col, target_col],
    inplace=True
)


# Text cleaning function

def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df[text_col] = df[text_col].apply(clean_text)

# Remove empty comments after cleaning
df = df[df[text_col] != ""]

print("\nData Cleaning Completed!")

print("Total Records After Cleaning:", len(df))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:",
      df.duplicated(subset=[text_col, target_col]).sum())


# ==================== 5. SENTIMENT DISTRIBUTION ====================

print("\n")
print("=" * 65)
print("SENTIMENT DISTRIBUTION")
print("=" * 65)

sentiment_counts = df[target_col].value_counts()

print(sentiment_counts)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target_col,
    hue=target_col,
    order=sentiment_counts.index,
    legend=False
)

plt.title("YouTube Comment Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ==================== 6. INPUT AND TARGET ====================

X = df[text_col]

y = df[target_col]

print("\nInput Feature: CommentText")
print("Target Variable: Sentiment")

print("\nNumber of Sentiment Classes:", y.nunique())

print("Sentiment Classes:", sorted(y.unique()))


# ==================== 7. TRAIN-TEST SPLIT ====================

print("\n")
print("=" * 65)
print("TRAIN TEST SPLITTING")
print("=" * 65)

class_counts = y.value_counts()

if len(class_counts) < 2 or class_counts.min() < 2:

    raise ValueError(
        "At least two sentiment classes are required, "
        "with at least two records per class."
    )

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)

print("\nTraining Samples:", len(X_train))

print("Testing Samples:", len(X_test))

print("Training Percentage: 80%")

print("Testing Percentage: 20%")


# ==================== 8. DEFINE ML ALGORITHMS ====================

print("\n")
print("=" * 65)
print("MACHINE LEARNING MODEL INITIALIZATION")
print("=" * 65)


models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "SVM": LinearSVC(
        max_iter=5000,
        random_state=42
    )

}


# ==================== 9. MODEL TRAINING ====================

trained_models = {}

results = []

confusion_matrices = {}

labels = sorted(y.unique())


for model_name, algorithm in models.items():

    print("\n")
    print("=" * 65)

    print("TRAINING MODEL:", model_name)

    print("=" * 65)

    # TF-IDF + Machine Learning Pipeline

    pipeline = Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                max_features=10000,

                ngram_range=(1, 2),

                min_df=1,

                sublinear_tf=True

            )
        ),

        (
            "classifier",

            algorithm
        )

    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict test data
    y_pred = pipeline.predict(X_test)

    # Evaluation metrics

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(

        y_test,
        y_pred,
        average="macro",
        zero_division=0

    )

    recall = recall_score(

        y_test,
        y_pred,
        average="macro",
        zero_division=0

    )

    f1 = f1_score(

        y_test,
        y_pred,
        average="macro",
        zero_division=0

    )

    # Store results

    results.append({

        "Algorithm": model_name,

        "Accuracy": accuracy * 100,

        "Precision": precision * 100,

        "Recall": recall * 100,

        "F1-Score": f1 * 100

    })

    trained_models[model_name] = pipeline

    confusion_matrices[model_name] = confusion_matrix(

        y_test,
        y_pred,
        labels=labels

    )

    # Print classification report

    print("\nClassification Report:")

    print(

        classification_report(

            y_test,
            y_pred,
            zero_division=0

        )

    )

    print("Accuracy: {:.2f}%".format(accuracy * 100))

    print("Precision: {:.2f}%".format(precision * 100))

    print("Recall: {:.2f}%".format(recall * 100))

    print("F1-Score: {:.2f}%".format(f1 * 100))


# ==================== 10. MODEL COMPARISON ====================

print("\n")
print("=" * 65)

print("FINAL MODEL COMPARISON")

print("=" * 65)


results_df = pd.DataFrame(results)

print(results_df.round(2).to_string(index=False))


# Find best model using Macro F1-score

best_model_name = results_df.loc[
    results_df["F1-Score"].idxmax(),
    "Algorithm"
]

best_model = trained_models[best_model_name]

print("\n")
print("BEST PERFORMING MODEL:", best_model_name)


# ==================== 11. PERFORMANCE GRAPH ====================

results_melted = results_df.melt(

    id_vars="Algorithm",

    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],

    var_name="Metric",

    value_name="Score"

)


plt.figure(figsize=(11, 6))

sns.barplot(

    data=results_melted,

    x="Algorithm",

    y="Score",

    hue="Metric"

)

plt.title("Comparison of Machine Learning Algorithms")

plt.xlabel("Machine Learning Algorithms")

plt.ylabel("Performance Score (%)")

plt.ylim(0, 105)

plt.legend(title="Evaluation Metrics")

plt.tight_layout()

plt.show()


# ==================== 12. CONFUSION MATRICES ====================

fig, axes = plt.subplots(

    1,
    3,
    figsize=(16, 5)

)


for ax, model_name in zip(axes, models.keys()):

    sns.heatmap(

        confusion_matrices[model_name],

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=labels,

        yticklabels=labels,

        ax=ax

    )

    ax.set_title(model_name)

    ax.set_xlabel("Predicted Sentiment")

    ax.set_ylabel("Actual Sentiment")


plt.suptitle("Confusion Matrix Comparison")

plt.tight_layout()

plt.show()


# ==================== 13. SENTIMENT PIE CHART ====================

plt.figure(figsize=(7, 7))

plt.pie(

    sentiment_counts.values,

    labels=sentiment_counts.index,

    autopct="%1.1f%%",

    startangle=90

)

plt.title("Overall YouTube Comments Sentiment")

plt.tight_layout()

plt.show()


# ==================== 14. NEW COMMENT PREDICTION ====================

print("\n")
print("=" * 65)

print("YOUTUBE COMMENT SENTIMENT PREDICTION")

print("=" * 65)

print("\nSelected Model:", best_model_name)

print("\nEnter a comment to predict its sentiment.")

print("Type 'exit' to stop prediction.")


while True:

    new_comment = input(

        "\nEnter YouTube Comment: "

    )

    if new_comment.strip().lower() == "exit":

        print("\nPrediction Session Completed!")

        break

    if not new_comment.strip():

        print("Please enter a valid comment.")

        continue

    cleaned_comment = clean_text(new_comment)

    prediction = best_model.predict(

        [cleaned_comment]

    )[0]

    print("\nOriginal Comment:", new_comment)

    print("Predicted Sentiment:", prediction)

    print("-" * 65)


# ==================== 15. PROGRAM COMPLETED ====================

print("\n")
print("=" * 65)

print("YOUTUBE SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")

print("=" * 65)

print("\nNo output files or plots were saved.")
print("All results were displayed inside Spyder.")
