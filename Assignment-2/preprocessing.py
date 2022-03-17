from random import random
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('../Assignment-1/data-jumlah-armada-bus-sekolah-2017.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # menghilangkan Missing Value Numeric dengan Mean data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:, 6])
# X[:, 6] = imputer.transform(X[:, 6])

# Encoding data kategori Atribut
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)
# print(y)