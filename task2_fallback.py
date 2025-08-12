# task2_fallback.py
import os
import io
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# If requests is available we'll use it to download raw CSVs
try:
    import requests
except ImportError:
    requests = None

FILENAME = "Mall_Customers.csv"
FALLBACK_URLS = [
    # public raw CSV mirrors that people commonly use (may change over time)
    "https://raw.githubusercontent.com/Mounika-Kajjam/Datasets/master/Mall_Customers.csv",
    "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv",
    # add other raw CSV URLs you trust here
]

def load_dataset(local_filename=FILENAME, urls=FALLBACK_URLS):
    # 1) try local file
    if os.path.exists(local_filename):
        print(f"Loading dataset from local file: {local_filename}")
        return pd.read_csv(local_filename)

    # 2) try fallback URLs (requires requests)
    if requests is None:
        print("Python package 'requests' not installed — skipping remote download attempts.")
    else:
        for url in urls:
            try:
                print(f"Trying to download dataset from: {url}")
                r = requests.get(url, timeout=10)
                if r.status_code == 200 and r.text.strip():
                    print("Downloaded dataset from URL.")
                    return pd.read_csv(io.StringIO(r.text))
                else:
                    print(f"URL returned status {r.status_code} or empty content.")
            except requests.RequestException as e:
                print(f"Failed to fetch {url}: {e}")

    # 3) fail with helpful instructions
    raise FileNotFoundError(
        f"Could not load '{local_filename}' from local disk or fallback URLs.\n"
        "Please download the 'Mall_Customers.csv' dataset and place it in the same folder as this script.\n"
        "Kaggle page (official source): https://www.kaggle.com/datasets/shwetabh123/mall-customers\n"
        "Or download from a trusted GitHub repo and update FALLBACK_URLS in this script."
    )

# -------------------------
# Main workflow
# -------------------------
try:
    df = load_dataset()
except Exception as e:
    print("ERROR loading dataset:\n", e)
    sys.exit(1)

print("First 5 rows:")
print(df.head())

# Select features (ensure correct column names)
required_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
if not all(col in df.columns for col in required_cols):
    print("Dataframe does not contain the expected columns. Columns found:", df.columns.tolist())
    sys.exit(1)

X = df[required_cols].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to inspect WCSS
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters k")
plt.ylabel("WCSS")
plt.show()

# Fit KMeans (choose k after inspecting elbow)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# Convert centers back to original scale (avoids the Series/array mismatch)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
for c in range(optimal_k):
    mask = df['Cluster'] == c
    plt.scatter(df.loc[mask, required_cols[0]], df.loc[mask, required_cols[1]], label=f"Cluster {c}", s=30)
plt.scatter(centers_original[:,0], centers_original[:,1], marker='X', s=200, color='k', label='Centroids')
plt.xlabel(required_cols[0])
plt.ylabel(required_cols[1])
plt.title("Customer Segments (K-Means)")
plt.legend()
plt.show()

# Silhouette (only if more than 1 cluster)
try:
    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score (KMeans, k={optimal_k}): {sil:.3f}")
except Exception as e:
    print("Couldn't compute silhouette score:", e)

# Bonus: DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = db_labels

plt.figure(figsize=(8,6))
plt.scatter(df[required_cols[0]], df[required_cols[1]], c=db_labels, cmap='tab10', s=30)
plt.xlabel(required_cols[0])
plt.ylabel(required_cols[1])
plt.title("DBSCAN Clustering (noise = -1)")
plt.show()

print("DBSCAN unique clusters:", sorted(set(db_labels)))
