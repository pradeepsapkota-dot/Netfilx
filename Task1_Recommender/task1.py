import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Loader

# 1. Load the data
data = Loader.load_data('Dataset/Dataset.csv')


# 2. Handle missing values
features = ["director", "country", "rating", "listed_in"]
for feature in features:
    data[feature] = data[feature].fillna('')

# 3. Create Tags
data['tags'] = data['director'] + ' ' + data['country'] + \
    ' ' + data['rating'] + ' ' + data['listed_in']

# 4. Convert Text into Numbers
vector = TfidfVectorizer(stop_words='english')
feature_vectors = vector.fit_transform(data['tags'])

# 5. Calculate Cosine Similarity
similarity = cosine_similarity(feature_vectors)

# 6. Build the recommendation function


def recommend(search_title, num_recommendations=5):
    if search_title not in data['title'].values:
        print(f"Error: '{search_title}' not found in the dataset.")
        return

    # FIX 2: Indent all of this so it runs INSIDE the function
    index = data[data['title'] == search_title].index[0]

    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[
        1:num_recommendations+1]

    print(f"Because you watched '{search_title}':\n")
    for i in movies_list:
        movie_idx = i[0]
        sim_score = i[1]
        print(
            f"- {data.iloc[movie_idx]['title']} (Similarity Score: {sim_score:.2f})")


# 7. System testing
recommend('Midnight Mass')
print('\n' + '='*40 + '\n')
recommend('Sankofa')
