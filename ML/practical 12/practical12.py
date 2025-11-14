import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


df = pd.read_csv("sales_data_sample.csv", encoding='latin1')
print("Dataset Loaded Successfully")
print(df.head())
print(df.info())



df_numeric = df.select_dtypes(include=[np.number])

df_numeric = df_numeric.fillna(df_numeric.mean())

print("\nNumerical columns selected for clustering:")
print(df_numeric.columns)

scaler = StandardScaler()
X = scaler.fit_transform(df_numeric)

print("\nData Standardized Successfully")
print("Shape of data:", X.shape)


wcss = []

for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()


optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

df['Cluster'] = y_kmeans
print("\n K-Means Clustering Completed with", optimal_k, "clusters")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette='Set2')
plt.title("K-Means Clustering Visualization (First Two Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title='Cluster')
plt.show()


linked = linkage(X, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()