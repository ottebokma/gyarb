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
#%matplotlib inline
plt.style.use("seaborn-whitegrid")

#loading data
data = pd.read_csv("data/journal_pone_0169490_s010.csv")

data.columns = ["patientnum", "marshall", "rotterdam", "ct_intracranial_final", "skullfx", "skullbasefx", "facialfx", "edh_final", "sdh_final", "sah_final", "contusion_final", "midlineshift_final", "cisterncomp_final", "mr_result", "gose_overallscore3m", "gose_overallscore6m", "ptsd_6m", "wais_psi_composite_6mo", "cvltshortdelaycuedrecallstandardscore_6mo", "cvltlongdelaycuedrecallstandardscore_6mo", "rs4680", "rs6277", "rs3219119", "rs11604671", "rs4938016", "rs1800497"]

X = data.loc[:, ("marshall","rotterdam","ct_intracranial_final", "skullfx", "skullbasefx","facialfx","edh_final", "sdh_final", "sah_final", "contusion_final", "midlineshift_final", "cisterncomp_final")].values
y = data.iloc[:, (15)].values

#generating dendrogram
Z = linkage(X, "ward")
dendrogram(Z, truncate_mode = "lastp", p=12, leaf_rotation = 45., leaf_font_size = 15., show_contracted = True)
plt.title("Agglomerative clustering dendrigram")
plt.xlabel("cluster size")
plt.ylabel("distance")

plt.axhline(y=25)
plt.axhline(y=15)

plt.show()
