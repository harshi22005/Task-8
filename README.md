🧮 K-Means Clustering — Task 8
📌 Project Overview

This project implements K-Means Clustering, an unsupervised machine learning technique used to group similar data points.
The goal is to segment customers based on purchasing behavior using the Mall Customer Segmentation Dataset.

This project demonstrates:

Applying K-Means

Finding optimal clusters using Elbow Method

Visualizing clusters using Matplotlib

Evaluating model performance using Silhouette Score

📁 Dataset Information

📂 Dataset Used: Mall Customer Segmentation Dataset
Place your dataset here:

data/Mall_Customers.csv

🧠 Concepts Covered

Unsupervised Learning

K-Means Clustering

WCSS (Within-Cluster Sum of Squares)

Elbow Method

Silhouette Score

PCA for 2D visualization

🛠️ Tools & Technologies Used
Library	Purpose
Pandas	Data handling
NumPy	Numerical operations
Scikit-learn	KMeans, Silhouette Score, PCA
Matplotlib	Graph plotting
Seaborn	Optional visualization styling

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn

📦 Project Folder Structure
📦 Task-8-KMeans-Clustering
│
├── data
│   └── Mall_Customers.csv
│
├── images
│   ├── elbow_method.png
│   ├── clusters_visualization.png
│
├── kmeans_clustering.py
├── README.md
└── requirements.txt

▶️ How to Run
1. Add Dataset

Place Mall_Customers.csv in the data folder.

2. Execute Script
python kmeans_clustering.py

3. Output Includes

Elbow Method Plot

Cluster Visualization Plot

Cluster-assigned DataFrame

Silhouette Score

📊 Elbow Method

The Elbow Method helps choose an optimal number of clusters by plotting WCSS values for different K.
The “bend” or “elbow” indicates the appropriate number of clusters.

Generated file:

images/elbow_method.png

🎨 Cluster Visualization

Clusters are visualized in 2D space after PCA reduction.
Each color represents one cluster, and centroids are marked separately.

Saved as:

images/clusters_visualization.png

🧪 Evaluation
✔ Silhouette Score

Indicates how well-separated and structured the clusters are.

Interpretation:

>0.5 → Good clustering

0 to 0.5 → Moderate

<0 → Poor (clusters overlapping)

🧑‍🎓 Author

 G Harshitha
AIML engineering student
