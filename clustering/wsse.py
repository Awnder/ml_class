import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import random

(X, y) = make_blobs(n_samples=500, n_features=2, cluster_std=0.6, centers=random.randrange(5, 15))

def calculate_wsse(X) -> np.ndarray:
    results = np.zeros((15-5,2))
    
    j = 0
    for i in range(5, 15):
        km = KMeans(n_clusters=i, max_iter=10)
        km.fit(X)
        results[j] = [i, km.inertia_]
        j += 1
    
    return results

def calculate_wsse_slopes(results: np.ndarray, threshold: float=100.0) -> int:
    ''' 
    gets decreasing slopes of wsse scores and finds the index of the elbow point below threshold, if any
    Parameters:
        results: np.ndarray of wsse indecies and scores
        threshold: float, tolerance difference between slopes to categorize as elbow
    Returns:
        float: the index of the elbow point below threshold, or the index of max value of all slopes
    '''
    slopes = []

    for i in range(0, len(results)-1):
        slope = results[i+1][1] - results[i][1]
        slopes.append(slope)

    for i in range(1, len(slopes)):
        # threshold slope difference to categorize as elbow
        if abs(slopes[i-1]) - abs(slopes[i]) > threshold:
            return results[i]

    # a guess if there is no elbow point over threshold
    return results[1]

results = calculate_wsse(X)
cluster_count = calculate_wsse_slopes(results)
print(cluster_count)

km = KMeans(n_clusters=cluster_count[0].astype(np.int32), max_iter=20)
clusters = km.fit_predict(X)
centroids = km.cluster_centers_

f, axis = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axis[0]
ax2 = axis[1]
ax3 = axis[2]

ax1.set_title("Original Data")
ax1.scatter(X[:,0], X[:,1], c=y)

ax2.set_title("WSSE Elbow")
ax2.plot(results[:,0], results[:,1])

ax3.set_title(f"Labeled Data (clusters: {cluster_count[0].astype(np.int32)})")
ax3.scatter(x=X[:,0], y=X[:,1], c=clusters)
ax3.scatter(x=centroids[:,0], y=centroids[:,1], marker="+", c='black')

plt.show()