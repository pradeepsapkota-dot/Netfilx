import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import Loader

data = Loader.load_data('Dataset/Dataset.csv')
categorized = ["rating"]
numeric_features = ["release_year", "year_added"]
X = data[categorized + numeric_features]
y = data["type"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorized)]
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classify", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Pipeline Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nDetailed Report\n: {classification_report(y_test, y_pred)}")
