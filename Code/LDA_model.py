
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
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

# Dimensionality Reduction using LDA
np.random.seed(42)

# Perform LDA
lda_model = LatentDirichletAllocation(n_components=5, learning_method='batch', random_state=42)
user_lda_scores = lda_model.fit_transform(user_like_matrix)

# Extract document-topic distributions (gamma)
gamma = user_lda_scores

# Extract topic-word distributions (beta)
beta = np.exp(lda_model.components_)

# Log likelihood
log_likelihood = lda_model.score(user_like_matrix)
print(log_likelihood)

# Printing 10 likes with highest LDA scores
for i in range(5):
    f = np.argsort(beta[i, :])
    temp = f[-15:]
    print(temp)
    print(likes.iloc[temp])



# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_result = kmeans.fit_predict(user_lda_scores)

# Plot k-means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=user_lda_scores[:, 0], y=user_lda_scores[:, 1], hue=kmeans_result, palette="Set2", legend="full")
plt.title("K-means Clustering")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend(title="Cluster")
plt.savefig("1.png")

# DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=10)
dbscan_result = dbscan.fit_predict(user_lda_scores)

# Plot DBSCAN clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=user_lda_scores[:, 0], y=user_lda_scores[:, 1], hue=dbscan_result, palette="Set2", legend="full")
plt.title("DBSCAN Clustering")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend(title="Cluster")
plt.savefig("2.png")

# # DBSCAN clustering with 5 clusters
# dbscan = DBSCAN(eps=0.05, min_samples=10)
# dbscan_result = dbscan.fit_predict(user_lda_scores)

# # Select only the first 5 clusters
# unique_labels = np.unique(dbscan_result)
# selected_labels = unique_labels[unique_labels != -1][:5]

# # Assign labels to points outside the selected clusters as -1 (noise)
# dbscan_result_selected = np.where(np.isin(dbscan_result, selected_labels), dbscan_result, -1)

# # Plot DBSCAN clusters with only 5 clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=user_lda_scores[:, 0], y=user_lda_scores[:, 1], hue=dbscan_result_selected, palette="Set2", legend="full")
# plt.title("DBSCAN Clustering (5 Clusters)")
# plt.xlabel("LDA Component 1")
# plt.ylabel("LDA Component 2")
# plt.legend(title="Cluster")
# plt.show()





# Printing 10 likes with highest LDA scores






