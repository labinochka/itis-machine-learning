import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def centr(X, k):
    centroids = []

    centroid_id = np.random.choice(X.shape[0])
    centroids.append(X[centroid_id])

    for _ in range(k - 1):
        dists = []
        for x in X:
            distances = np.linalg.norm(centroids - x, axis=1)
            dists.append(distances.min())

        centroid_ind = np.array(dists).argmax()
        next_centroid = X[centroid_ind]
        centroids.append(next_centroid)

    centroids = np.array(centroids)
    return centroids


iris = load_iris()
X = iris.data

k = 3
centroids = centr(X, k)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='gray')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('Initial state')
plt.show()

for i in range(100):

    clusters = {i: [] for i in range(k)}

    for x in X:
        distances = np.linalg.norm(centroids - x, axis=1)
        cluster_ind = distances.argmin()
        clusters[cluster_ind].append(x)

    new_centroids = {}
    for c in clusters:
        new_centroids[c] = np.mean(clusters[c], axis=0)

    plt.figure()
    colors = ['blue', 'green', 'pink', 'purple']
    for c in clusters:
        for x in clusters[c]:
            plt.scatter(x[0], x[1], color=colors[c])

    for c in new_centroids:
        plt.scatter(new_centroids[c][0], new_centroids[c][1], c='red', marker='x')

    plt.title(f'Step {i + 1}')
    plt.show()

    is_stop = False
    for clust in range(len(centroids)):
        if np.linalg.norm(centroids[clust] - new_centroids[clust]) <= 1e-4:
            is_stop = True
            break
    if is_stop:
        print(f'Stopping at iter {i + 1}')
        break
    new_centroids = dict(sorted(new_centroids.items()))
    new_centroids = np.array(list(new_centroids.values()))
    centroids = new_centroids.copy()
