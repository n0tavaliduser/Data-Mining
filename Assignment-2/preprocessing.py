from importlib import import_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../Assignment-1/data-jumlah-armada-bus-sekolah-2017.csv')
# print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -3].values

print("data X : ", X)
# print("data Y : ", y)

# Melakukan split dataset menjadi Training set dan Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Menghilangkan missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X)
X = imputer.transform(X)
# print(imputer.transform(X))

# Encoding data kategori
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 3, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Melakukan split dataset menjadi Training set dan Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("X_train : ", X_train)
print("X_test : ", X_test)
print("y_train : ", y_train)
print("y_test : ", y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:] = sc.fit_transform(X_train[:, 5:].reshape(1, -1))
X_test[:, 5:] = sc.fit_transform(X_test[:, 5:].reshape(1, -1))
print(X_test)
print(X_train)