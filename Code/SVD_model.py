
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
users = pd.read_csv("/home/local/ASUAD/qlin36/CSE-573-Group-21-Project-21-Personality-classification-with-Social-Media/Data/users.csv")
likes = pd.read_csv("/home/local/ASUAD/qlin36/CSE-573-Group-21-Project-21-Personality-classification-with-Social-Media/Data/likes.csv")
user_likes = pd.read_csv("/home/local/ASUAD/qlin36/CSE-573-Group-21-Project-21-Personality-classification-with-Social-Media/Data/users-likes.csv")

# Merge user_likes with users and likes
user_likes = pd.merge(user_likes, users, on="userid")
user_likes = pd.merge(user_likes, likes, on="likeid")

# Count occurrences of each user and like
user_counts = user_likes["userid"].value_counts()
like_counts = user_likes["likeid"].value_counts()

# Identify sparse users and likes
sparse_users = user_counts[user_counts < 50].index
sparse_likes = like_counts[like_counts < 150].index

# Remove sparse users and likes
user_likes = user_likes[~user_likes["userid"].isin(sparse_users)]
user_likes = user_likes[~user_likes["likeid"].isin(sparse_likes)]

print(user_likes.head(10))

# Get the list of user IDs present in user_likes after removing sparse entries
users_in_user_likes = user_likes["userid"].unique()
likes_in_user_likes = user_likes["likeid"].unique()

# Filter out sparse users from users DataFrame
users = users[users["userid"].isin(users_in_user_likes)]
likes = likes[likes["likeid"].isin(likes_in_user_likes)]

# Create a sparse user-like matrix
user_like_matrix = csr_matrix((np.ones(len(user_likes)), 
                               (user_likes["userid"].astype("category").cat.codes.values,
                                user_likes["likeid"].astype("category").cat.codes.values)))

# Perform SVD
svd_model = TruncatedSVD(n_components=5, random_state=42)
user_svd_scores = svd_model.fit_transform(user_like_matrix)

# Get the components of the SVD model
svd_components = svd_model.components_

# Get the indices of the top and bottom likes based on the SVD scores along each dimension
top_likes_indices = [np.argsort(component)[-10:][::-1] for component in svd_components]
bottom_likes_indices = [np.argsort(component)[:10] for component in svd_components]

# Retrieve the top and bottom likes from the like names
for i in range(5):
    top_likes_indice = top_likes_indices[i]
    print(top_likes_indice)
    print(likes.iloc[top_likes_indice])

for i in range(5):
    bottom_likes_indice = bottom_likes_indices[i]
    print(bottom_likes_indice)
    print(likes.iloc[bottom_likes_indice]) 

# K-means clustering with SVD scores
kmeans_svd = KMeans(n_clusters=5, random_state=42)
kmeans_svd_result = kmeans_svd.fit_predict(user_svd_scores)



# Plot k-means clusters with SVD scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x=user_svd_scores[:, 0], y=user_svd_scores[:, 1], hue=kmeans_svd_result, palette="Set2", legend="full")
plt.title("K-means Clustering with SVD Scores")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend(title="Cluster")
plt.savefig("kmeans_svd.png")

# DBSCAN clustering with SVD scores
dbscan_svd = DBSCAN(eps=0.1, min_samples=10)
dbscan_svd_result = dbscan_svd.fit_predict(user_svd_scores)

# Plot DBSCAN clusters with SVD scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x=user_svd_scores[:, 0], y=user_svd_scores[:, 1], hue=dbscan_svd_result, palette="Set2", legend="full")
plt.title("DBSCAN Clustering with SVD Scores")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend(title="Cluster")
plt.savefig("dbscan_svd.png")







