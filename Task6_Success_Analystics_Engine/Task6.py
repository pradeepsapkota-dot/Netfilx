# Task6 Success Analytics Engine
import Loader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # to hide unnecessary math warning.

# 1. Load the data
data = Loader.load_data('Dataset/Dataset.csv')

# Dropping missing countries for accuracy
df = data.dropna(subset=['country']).copy()
# Creating Business Target: International (1) and US (0)content
df['is_international'] = df['country'].apply(
    lambda x: 0 if 'United States' in str(x) else 1)


# Advance Text Feature Engineering
print("Vectorizing content genres...")
tfidf = TfidfVectorizer(max_features=40)
genre_matrix = tfidf.fit_transform(df['listed_in']).toarray()
genre_cols = tfidf.get_feature_names_out()

# Combine NLP features with numerical features
X = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
X['release_year'] = df['release_year']
X['duration_no'] = df['duration_no'].fillna(df['duration_no'].median())

y = df['is_international']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\n--- Step 2 and 3: Model Building and Comparison ---")
# Building a pipeline of multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
best_model_name = ""
best_accuracy = 0

# Train and compare
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name

print(f"\n>> WINNER : {best_model_name} with {best_accuracy:.2%} accuracy! <<")
print("\n--- Step 4: Automated Insights ---")
# Extract feature importance
best_model = models[best_model_name]

# HAndel different model attribute types for importance extraction
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nAutomated Business Intelligence: ")
print("Top 5 Drivers for Netflix International Content: ")
for i, row in importances_df.head(5).iterrows():
    print(f"- {row['Feature'].capitalize()}: {row['Importance']:.4f}")

print("\n--- Step 5: Visual Reports ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Plot 1: Model Comparison
sns.barplot(x=list(results.keys()), y=list(
    results.values()), ax=axes[0], color="royalblue")
axes[0].set_title('Model Performance Comparison')
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel('Accuracy Score')

# Plot 2: Feature Importance (Business Insights)
sns.barplot(x='Importance', y='Feature', data=importances_df.head(
    10), ax=axes[1], color="darkorange")
axes[1].set_title(f'Top 10 International Growth Drivers ({best_model_name})')
axes[1].set_xlabel('Predictive Importance')

plt.tight_layout()
plt.show()
