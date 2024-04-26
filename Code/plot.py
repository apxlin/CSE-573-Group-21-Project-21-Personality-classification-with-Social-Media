
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the users DataFrame with the clustering results
users = pd.read_csv("/home/local/ASUAD/qlin36/CSE-573-Group-21-Project-21-Personality-classification-with-Social-Media/Data/filtered_users.csv")

# Set the style of seaborn plots
sns.set(style="whitegrid")

# Plotting loop for each property
properties = ['gender', 'age', 'political', 'ope', 'con', 'ext', 'agr', 'neu']
for prop in properties:
    # Create a new figure
    plt.figure(figsize=(15, 6))
    
    # Plot each property across clusters
    for cluster in range(5):  # Assuming 5 clusters
        # Filter users based on the cluster
        cluster_data = users[users['kmeans_svd_cluster'] == cluster]
        
        # Plot the property distribution
        plt.subplot(1, 5, cluster + 1)
        if prop == 'age':
            sns.histplot(data=cluster_data, x=prop, bins=10, kde=True)
        elif prop in ['ope', 'con', 'ext', 'agr', 'neu']:
            sns.histplot(data=cluster_data, x=prop, kde=True)
        else:
            sns.countplot(x=prop, data=cluster_data)
        plt.title(f'Cluster {cluster} {prop.capitalize()} Distribution')

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"/home/local/ASUAD/qlin36/CSE-573-Group-21-Project-21-Personality-classification-with-Social-Media/Evaluations/{prop}_distribution_across_clusters.png")
    
    # Close the current figure to release memory
    plt.close()