import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from w_kmeans import KMeans

X, y = make_blobs(centers = 3, n_samples = 500, n_features = 3, shuffle = True)
print (X.shape)

clusters = len(np.unique(y))
print (clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=False)
y_pred = k.predict(X)

k.plot()