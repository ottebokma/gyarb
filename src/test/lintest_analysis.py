import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("data/journal_pone_0169490_s010.csv")
data = data[["marshall","rotterdam","ptsd_6mo"]]

#print(data.head())
#print (data.shape)
#print (data.isnull().tail())
#print (data.isnull().sum())
#print (data[data.ptsd_6mo.isnull()])

data = data.dropna(subset=["ptsd_6mo"],how = "all")

predict = "ptsd_6mo"

attributes = np.array(data.drop([predict], 1))
labels = np.array(data[predict])

attrib_train, attrib_test, label_train, label_test = sklearn.model_selection.train_test_split(attributes, labels, test_size = 0.1)

best = 0

for _ in range (10):
    attrib_train, attrib_test, label_train, label_test = sklearn.model_selection.train_test_split(attributes, labels, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(attrib_train, label_train)
    
    acc = linear.score(attrib_test, label_test)
    print ("accuracy:" , acc)
    
    if acc > best:
        best = acc
        with open("lintestmod.pickle", "wb") as f:
            pickle.dump(linear, f)

print ("best:" , best)

print ("coefficient:" , linear.coef_)
print ("intercept:" , linear.intercept_)

        
pickle_open = open("lintestmod.pickle", "rb")
linear = pickle.load(pickle_open)

predictions = linear.predict(attrib_test)

for x in range(len(predictions)):
    print (predictions[x], attrib_test[x], label_test[x])

x = "rotterdam"
    
style.use("ggplot")
pyplot.scatter(data[x],data[predict])
pyplot.xlabel(x)
pyplot.ylabel(predict)
pyplot.show()
