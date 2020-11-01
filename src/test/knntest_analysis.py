import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("data/journal_pone_0169490_s010.csv")
data = data[["marshall","rotterdam","skullfx","skullbasefx","facialfx","edh_final","sdh_final","sah_final","gose_overallscore3m"]]

data = data.dropna(subset=["gose_overallscore3m"],how = "all")

prep = preprocessing.LabelEncoder()
marshall = prep.fit_transform(list(data["marshall"]))
rotterdam = prep.fit_transform(list(data["rotterdam"]))
skullfx = prep.fit_transform(list(data["skullfx"]))
skullbasefx = prep.fit_transform(list(data["skullbasefx"]))
facialfx = prep.fit_transform(list(data["facialfx"]))
edh_final = prep.fit_transform(list(data["edh_final"]))
sdh_final = prep.fit_transform(list(data["sdh_final"]))
sah_final = prep.fit_transform(list(data["sah_final"]))
gose = prep.fit_transform(list(data["gose_overallscore3m"]))

predict = "gose_overallscore3m"

attrib = list(zip(marshall, rotterdam, skullfx, skullbasefx, facialfx, edh_final, sdh_final, sah_final))
label = list(gose)

attrib_train, attrib_test, label_train, label_test = sklearn.model_selection.train_test_split(attrib, label, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=11)
model.fit(attrib_train, label_train)

acc = model.score(attrib_test, label_test)
predicted = model.predict(attrib_test)
print (acc)

for x in range(len(predicted)):
    print ("predicted:", predicted[x], "data:", attrib_test[x], "actual:", label_test[x])
