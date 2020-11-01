import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import load_digits



data = pd.read_csv("data/journal_pone_0169490_s010.csv")
data = data[["marshall","rotterdam","skullfx","skullbasefx","facialfx","edh_final","sdh_final","sah_final","gose_overallscore3m"]]

#data = scale(digits.data)
data = data.dropna(subset=["gose_overallscore3m"],how = "all")

y = "gose_overallscore3"

k = 8

samples, features = data.shape
"""
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
"""
clf = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=500)

bench_k_means(clf, "1", data)
