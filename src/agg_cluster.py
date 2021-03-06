import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

np.set_printoptions(precision = 4, suppress = True)
plt.figure(figsize = (10,3))
plt.style.use("seaborn-whitegrid")

#choosing comparison data (y value)
cmp = "ptsd_6mo"

#loading data
data = pd.read_csv("data/journal_pone_0169490_s010.csv")

data.columns = ["patientnum", "marshall", "rotterdam", "ct_intracranial_final", "skullfx", "skullbasefx", "facialfx", "edh_final", "sdh_final", "sah_final", "contusion_final", "midlineshift_final", "cisterncomp_final", "mr_result", "gose_overallscore3m", "gose_overallscore6m", "ptsd_6mo", "wais_psi_composite_6mo", "cvltshortdelaycuedrecallstandardscore_6mo", "cvltlongdelaycuedrecallstandardscore_6mo", "rs4680", "rs6277", "rs3219119", "rs11604671", "rs4938016", "rs1800497"]

data = data.dropna(how="any")

X = data.loc[:, ("marshall", "rotterdam", "ct_intracranial_final", "skullfx", "skullbasefx", "facialfx", "edh_final", "sdh_final", "sah_final", "contusion_final", "midlineshift_final", "cisterncomp_final", "rs4680", "rs6277", "rs3219119", "rs11604671", "rs4938016", "rs1800497")].values
y = data.loc[:, (cmp)].values

y = y.astype(int)

#generating dendrogram
Z = linkage(X, "ward")
dendrogram(Z, truncate_mode = "lastp", p=12, leaf_rotation = 45., leaf_font_size = 15., show_contracted = True)
plt.title("Agglomerative Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")

plt.axhline(y=25)
plt.axhline(y=15)

plt.show()


#generating hierarchical clusers
k = 2

#ward linkage
Hclustering = AgglomerativeClustering(n_clusters = k, affinity="euclidean", linkage="ward")

Hclustering.fit(X)

labels = Hclustering.labels_
print("")
print (y)
#print (Hclustering.labels_)
print (labels)

print (sm.accuracy_score(y, labels))

#complete linkage
Hclustering = AgglomerativeClustering(n_clusters = k, affinity="euclidean", linkage="complete")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="manhattan", linkage="complete")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="l1", linkage="complete")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="l2", linkage="complete")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="cosine", linkage="complete")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

#average linkage
Hclustering = AgglomerativeClustering(n_clusters = k, affinity="euclidean", linkage="average")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))


Hclustering = AgglomerativeClustering(n_clusters = k, affinity="manhattan", linkage="average")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="l1", linkage="average")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="l2", linkage="average")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="cosine", linkage="average")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

#single linkage
Hclustering = AgglomerativeClustering(n_clusters = k, affinity="euclidean", linkage="single")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))


Hclustering = AgglomerativeClustering(n_clusters = k, affinity="manhattan", linkage="single")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="l1", linkage="single")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="l2", linkage="single")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))

Hclustering = AgglomerativeClustering(n_clusters = k, affinity="cosine", linkage="single")

Hclustering.fit(X)
labels = Hclustering.labels_

print (sm.accuracy_score(y, labels))
