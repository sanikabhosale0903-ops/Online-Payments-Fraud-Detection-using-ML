import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib

# Load dataset
data = pd.read_csv("../data/PS_20174392719_1491204439457_log.csv")


# 🔥 IMPORTANT (speed fix)
data = data.sample(50000)

model = RandomForestClassifier(n_estimators=50, n_jobs=-1)

print("Columns:", data.columns)

# Drop unwanted columns
data.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

# Encode categorical data
le = LabelEncoder()
data["type"] = le.fit_transform(data["type"])

# Features & Target
X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
models = {
    "RandomForest": RandomForestClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "SVM": SVC()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(name, "Accuracy:", score)

    if score > best_score:
        best_score = score
        best_model = model

# Save best model
joblib.dump(best_model, "../fraud_model.pkl")

print("Best model saved successfully!")