#Task 4 Content Segmentation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import Loader

data = Loader.load_data('Dataset/Dataset.csv')



#Filter to Movies only
movies = data[data['type'] == "Movie"].copy()
movies = movies.dropna(subset=["release_year", "duration_no"]) 

#Select Numerical Features
X = movies[["release_year", "duration_no"]]

#Scaling the data ny K-Means calculate distances
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

# Applying the K Means Cluster to find 3 distinct group
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
movies["cluster"] = kmeans.fit_predict(X_scaled)

#Interpreting the Cluster
print("Average Values for Each Cluster")
print(movies.groupby('cluster')[["release_year", "duration_no"]].mean())

#Visualizing the Results
plt.figure(figsize=(10,6)) 
sns.scatterplot(x='release_year',
                y='duration_no',
                hue='cluster',
                data= movies,
                palette='viridis',
                alpha = 0.7)
plt.title('Netflix Movie Segmentation: Release Year vs Duration')
plt.xlabel('Release Year')
plt.ylabel('Duration (minutes)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

