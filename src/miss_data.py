import pandas as pd

data = pd.read_csv("data/journal_pone_0169490_s010.csv")

print(data.shape)

#data = data.dropna()

print(data.isnull().sum())

#print(data[data.gose_overallscore3m.isnull()])
