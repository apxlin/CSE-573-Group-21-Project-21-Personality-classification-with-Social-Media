import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
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

# Split users into 10 groups
np.random.seed(42)
folds = np.random.randint(1, 11, size=len(users))

# Take users from group 1 and assign them to the TEST subset
test = folds == 1

# Filter the user-like matrix for the training subset
train_user_like_matrix = user_like_matrix[~test]

# Choose which k are to be included in the analysis
keySet = list(range(2, 11)) + [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Preset an empty dictionary to hold the results
resultSet = {}

# Run the code below for each k in keySet
for k in keySet:
    # Perform TruncatedSVD
    svd_model = TruncatedSVD(n_components=k, random_state=42)
    svd = svd_model.fit_transform(train_user_like_matrix)  # Assuming M is your data matrix
    v_rot = svd_model.components_
    u_rot = user_like_matrix.dot(v_rot.T)
    # Fit linear regression model
    linear_model = LinearRegression()
    linear_model.fit(u_rot[~test], users.loc[~test, 'ope'])

    # Predict on the test subset
    predictions = linear_model.predict(u_rot[test])

    # Calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(users.loc[test, 'ope'], predictions)[0, 1]

    # Save the resulting correlation coefficient
    resultSet[str(k)] = correlation_coefficient

best_k = max(resultSet, key=resultSet.get)
best_correlation = resultSet[best_k]

# Build model to predict user traits
# Fit the SVD model on the training subset
best_k = int(best_k)
svd_model = TruncatedSVD(n_components=best_k, random_state=42)
Msvd = svd_model.fit_transform(train_user_like_matrix)
v_rot = svd_model.components_

# Rotate user SVD scores *for the entire sample*
u_rot = user_like_matrix.dot(v_rot.T)

# Calculate average prediction correlation
def calculate_average_prediction_correlation_svd(model, label):
    # Initialize an empty array to store results
    results = np.full(len(users), np.nan)
    
    # Check if the variable is dichotomous (contains only 2 unique values)
    unique_values = users[label].dropna().unique()
    if len(unique_values) == 2:
        # Fit logistic regression model
        model = LogisticRegression()
        model.fit(u_rot[~test], users.loc[~test, label])
        
        # Predict probabilities on the test subset
        probabilities = model.predict_proba(u_rot[test])[:, 1]  # Probability of positive class
        results[test] = probabilities
    else:
        # Fit linear regression model
        model = LinearRegression()
        model.fit(u_rot[~test], users.loc[~test, label])
        
        # Predict on the test subset
        predictions = model.predict(u_rot[test])
        results[test] = predictions
    
    print(f"Variable {label} done.")
    
    return results

# List of labels
labels = ['ope', 'con', 'ext', 'agr', 'neu']

# Calculate correlation for each label
for label in labels:
    # Calculate average prediction correlation
    predictions = calculate_average_prediction_correlation_svd(LinearRegression(), label)
    
    # Filter out NaN values from users data
    ground_truth = users.loc[test, label].dropna()
    predictions = predictions[~np.isnan(predictions)]
    
    # Ensure dimensions match before calculating correlation
    min_length = min(len(ground_truth), len(predictions))
    ground_truth = ground_truth[:min_length]
    predictions = predictions[:min_length]
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    
    print(f"Correlation for {label}: {correlation}")

# Calculate MSE and R2 for each linear model
linear_results = []

for label in labels:
    # Fit linear regression model
    linear_model.fit(u_rot[~test], users.loc[~test, label])
    
    # Predict on the test subset
    predictions = linear_model.predict(u_rot[test])
    
    # Calculate MSE and R2
    mse = mean_squared_error(users.loc[test, label], predictions)
    r2 = r2_score(users.loc[test, label], predictions)
    
    # Append results to list
    linear_results.append({'Label': label, 'MSE': mse, 'R2': r2})

# Convert results to DataFrame for better visualization
linear_results_df = pd.DataFrame(linear_results)

# Display results
print("\nLinear Regression Results:")
print(linear_results_df)



#LDA model
lda_model = LatentDirichletAllocation(n_components=5, learning_method='batch', random_state=42)
user_lda_scores = lda_model.fit_transform(train_user_like_matrix)

def calculate_average_prediction_correlation_lda(model, label):
    results = np.full(len(users), np.nan)
    
    # Check if the variable is dichotomous (contains only 2 unique values)
    unique_values = users[label].dropna().unique()
    if len(unique_values) == 2:
        # Fit logistic regression model
        model = LogisticRegression()
        model.fit(user_lda_scores[~test], users.loc[~test, label])
        
        # Predict probabilities on the test subset
        probabilities = model.predict_proba(user_lda_scores[test])[:, 1]  # Probability of positive class
        results[test] = probabilities
    else:
        # Fit linear regression model
        model = LinearRegression()
        model.fit(user_lda_scores[~test], users.loc[~test, label])
        
        # Predict on the test subset
        predictions = model.predict(user_lda_scores[test])
        results[test] = predictions
    
    print(f"Variable {label} done.")
    
    return results


# Initialize linear regression model
linear_model = LinearRegression()

# List of labels
labels = ['ope', 'con', 'ext', 'agr', 'neu']

# Calculate average prediction accuracy for each label
for label in labels:
    # Calculate average prediction correlation
    predictions = calculate_average_prediction_correlation_lda(LinearRegression(), label)
    
    # Filter out NaN values from users data
    ground_truth = users.loc[test, label].dropna()
    predictions = predictions[~np.isnan(predictions)]
    
    # Ensure dimensions match before calculating correlation
    min_length = min(len(ground_truth), len(predictions))
    ground_truth = ground_truth[:min_length]
    predictions = predictions[:min_length]
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    
    print(f"Correlation for {label}: {correlation}")


# Calculate MSE and R2 for each linear model
linear_results = []

for label in labels:
    # Fit linear regression model
    linear_model.fit(u_rot[~test], users.loc[~test, label])
    
    # Predict on the test subset
    predictions = linear_model.predict(u_rot[test])
    
    # Calculate MSE and R2
    mse = mean_squared_error(users.loc[test, label], predictions)
    r2 = r2_score(users.loc[test, label], predictions)
    
    # Append results to list
    linear_results.append({'Label': label, 'MSE': mse, 'R2': r2})

# Convert results to DataFrame for better visualization
linear_results_df = pd.DataFrame(linear_results)

# Display results
print("\nLinear Regression Results:")
print(linear_results_df)

