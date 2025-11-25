import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import zipfile

# -------------------------------
# 1. LOAD DATASET FROM ZIP FILE
# -------------------------------
zip_path = "C:\\Users\\G HARSHITHA\\Downloads\\archive (9).zip"  # UPDATE ONLY THIS IF NEEDED

with zipfile.ZipFile(zip_path) as z:
    with z.open("Mall_Customers.csv") as f:   # This file is inside the ZIP
        df = pd.read_csv(f)

print("Dataset Loaded Successfully!\n")
print(df.head())

# --------------------------------
# 2. SELECT FEATURES & SCALE THEM
# --------------------------------
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 3. ELBOW METHOD (FIND K)
# ---------------------------
inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title("Elbow Method to Find Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# ----------------------------------------------------
# 4. TRAIN FINAL K-MEANS MODEL (USUALLY K = 5)
# ----------------------------------------------------
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels

# -----------------------------
# 5. SILHOUETTE SCORE
# -----------------------------
score = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score: {score}")

# -----------------------------
# 6. PCA FOR 2D VISUALIZATION
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis")
plt.title("K-Means Clusters (PCA Visualization)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()
