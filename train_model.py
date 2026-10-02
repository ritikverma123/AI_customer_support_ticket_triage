import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "tickets.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["ticket_text", "category", "urgency"])
X = df["ticket_text"].astype(str)
y_category = df["category"]
y_urgency = df["urgency"]

# One common split so both models use the same held-out tickets.
indices = df.index
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y_category
)

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_cat_train, y_cat_test = y_category.loc[train_idx], y_category.loc[test_idx]
y_urg_train, y_urg_test = y_urgency.loc[train_idx], y_urgency.loc[test_idx]

def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

category_model = make_model()
urgency_model = make_model()

category_model.fit(X_train, y_cat_train)
urgency_model.fit(X_train, y_urg_train)

cat_pred = category_model.predict(X_test)
urg_pred = urgency_model.predict(X_test)

cat_f1 = f1_score(y_cat_test, cat_pred, average="weighted")
urg_f1 = f1_score(y_urg_test, urg_pred, average="weighted")

print(f"Dataset size: {len(df)} tickets")
print("\nCategory distribution:")
print(y_category.value_counts())
print("\nUrgency distribution:")
print(y_urgency.value_counts())

print("\n=== CATEGORY MODEL ===")
print(classification_report(y_cat_test, cat_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_cat_test, cat_pred))

print("\n=== URGENCY MODEL ===")
print(classification_report(y_urg_test, urg_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_urg_test, urg_pred))

joblib.dump(category_model, MODELS / "category_model.joblib")
joblib.dump(urgency_model, MODELS / "urgency_model.joblib")

metrics = {
    "dataset_size": int(len(df)),
    "test_size": int(len(test_idx)),
    "category_weighted_f1": round(float(cat_f1), 4),
    "urgency_weighted_f1": round(float(urg_f1), 4),
    "category_classes": list(category_model.classes_),
    "urgency_classes": list(urgency_model.classes_),
    "note": "Dataset is a larger synthetic demonstration dataset. Replace or supplement it with real/approved historical tickets for a production study."
}
(MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

print("\nModels saved in:", MODELS)
