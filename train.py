import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils import clean_text

# =========================
# 1. Load Dataset
# =========================
# Your CSV is tab-separated with 2 columns: label, message
df = pd.read_csv('spam.csv', sep='\t', names=['label', 'message'])

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =========================
# 2. Preprocessing
# =========================
df['message'] = df['message'].apply(clean_text)

# =========================
# 3. Feature Extraction
# =========================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label']

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Models
# =========================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}

best_model = None
best_score = 0

# =========================
# 6. Training & Evaluation
# =========================
for name, model in models.items():
    print(f"\n🔍 Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = model

# =========================
# 7. Save Model
# =========================
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Best Model Saved Successfully!")
print(f"🏆 Best Accuracy: {best_score:.4f}")