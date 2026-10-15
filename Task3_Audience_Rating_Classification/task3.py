import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import Loader

data = Loader.load_data('Dataset/Dataset.csv')
# Audience Rating classification

# grouping the ratings
adults = ['TV-MA', 'R', 'NC-17', 'UR', 'NR']
teen = ['TV-14', 'PG-13', 'TV-PG']
kids = ['TV-G', 'TV-Y', 'TV-Y7', 'PG', 'G', 'TV-Y7-FV']

# Function created to assign ratings


def categorize_rating(rating):
    if rating in adults:
        return 'Adults'
    elif rating in teen:
        return 'Teen'
    elif rating in kids:
        return 'Kids'
    else:
        return 'Unknown'


# Creating a new column to apply function
data['target_rating'] = data['rating'].apply(categorize_rating)

# Selecting feature and target
# using type and release_year as a clue
X = data[['type', 'release_year']]
y = data['target_rating']

# Converting text to numbers
X = pd.get_dummies(X, columns=['type'])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build and train the decision tree algorithm
# using max_depth = 5 to stop the tree from growing complex and difficult to memorize
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Test the model and print the report
predictions = model.predict(X_test)

print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
print(classification_report(y_test, predictions))
