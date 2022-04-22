from sklearn.tree import DecisionTreeClassifier
# from sklearn import datasets
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree

irisDataset = pd.read_csv("../Assignment-4/databus.csv", delimiter=',', header=0)

irisDataset['jenis_armada'] = pd.factorize(irisDataset.jenis_armada)[0]
# irisDataset = irisDataset.drop(labels="")
irisDataset = irisDataset.to_numpy()

dataTraining = np.concatenate((irisDataset[0:80,:],irisDataset[100:180,:]), axis=0)
dataTesting = np.concatenate((irisDataset[80:100,:],irisDataset[180:200,:]), axis=0)

inputTraining = dataTraining[:,0:4]
inputTesting = dataTesting[:,0:4]
labelTraining = dataTraining[:,4]
labelTesting = dataTesting[:,4]

model = tree.DecisionTreeClassifier()
model = model.fit(inputTraining, labelTraining)

hasilPrediksi = model.predict(inputTesting)
print("label sebenarnya = ", labelTesting)
print("hasil prediksi = ", hasilPrediksi)

prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi !=  labelTesting).sum()
print("prediksi benar = ", prediksiBenar, " data")
print("prediksi salah = ", prediksiSalah, " data")
print("akurasi = ", prediksiBenar/(prediksiBenar + prediksiSalah) * 100, "%")