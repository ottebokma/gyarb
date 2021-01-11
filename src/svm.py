import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


#choosing comparison data (y value)
#cmp = "ptsd_6mo"

#loading data
data = pd.read_csv("data/journal_pone_0169490_s010.csv")

""" all data
"patientnum", "marshall", "rotterdam", "ct_intracranial_final", "skullfx", "skullbasefx", "facialfx", "edh_final", "sdh_final", "sah_final", "contusion_final", "midlineshift_final", "cisterncomp_final", "mr_result", "gose_overallscore3m", "gose_overallscore6m", "ptsd_6mo", "wais_psi_composite_6mo", "cvltshortdelaycuedrecallstandardscore_6mo", "cvltlongdelaycuedrecallstandardscore_6mo", "rs4680", "rs6277", "rs3219119", "rs11604671", "rs4938016", "rs1800497
"""

data = data[["marshall", "rotterdam", "ct_intracranial_final", "skullfx", "skullbasefx", "facialfx", "edh_final", "sdh_final", "sah_final", "contusion_final", "midlineshift_final", "cisterncomp_final", "gose_overallscore6m"]]

predict = "gose_overallscore6m"

data = data.dropna(subset=[predict], how="any")

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
y = y.astype("int")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel="poly", degree = 5, C = 3)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

#print("")
#print (y_pred)
#print (y_test)

acc = metrics.accuracy_score(y_test, y_pred)
print (acc)

clf2 = KNeighborsClassifier(n_neighbors = 5)
clf2.fit(x_train, y_train)

y_pred2 = clf2.predict(x_test)

acc2 = metrics.accuracy_score(y_test, y_pred2)
print (acc2)
