# CSE-573-Group-21-Project-21-Personality-classification-with-Social-Media

Group Members:

Jiaxin Dai

Qi Lin

Prikanksha Maheshwari

Varshil Chetankumar Shah

Rahul Suddala

Dataset Link - https://drive.google.com/file/d/15v2umVbv1OarPfNrePH9V-KPk7qqA6Cg/edit

Reference Paper - https://psycnet.apa.org/fulltext/2016-57141-003.pdf
                - https://www.pnas.org/doi/full/10.1073/pnas.1218772110

Steps to Run:

Clone Repository to local system

Download three files (users.csv, users-likes.csv, likes.csv) from the given dataset link above and place the three files in the repo's Data folder.

Download and install Python 3.8 in your system

In your terminal, go to the Code folder and run the following code files:

a. Run command - "python3 SVD_model.py" for SVD dimensionality reduction and visualizing K-Means and DBSCAN algorithms for SVD clusters.

b. Run command - "python3 LDA_model" for LDA dimensionality reduction and visualizing K-Means and DBSCAN algorithms for LDA clusters.

c. Run command - "python3 Prediction_model" for performing Linear and Logistic Regression algorithms to predict user personality types.

The Evaluations folder contains the resulting Python visualization files based on the analysis results above.
