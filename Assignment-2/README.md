preprocessing.py
========================================

File python dapat dilihat di [*Python file*](https://github.com/n0tavaliduser/Data-Mining/blob/main/Assignment-2/preprocessing.py)

File dataset dapat dilihat di [dataset](https://github.com/n0tavaliduser/Data-Mining/blob/main/Assignment-1/data-jumlah-armada-bus-sekolah-2017.csv) tentang data jumlah armada bus sekolah tahun 2017, terdiri dari 8 kolom dan 71 record data.



Preprocessing Data
========================================

Pengolahan suatu data yang bersifat mentah dalam format tertentu untuk disempurnakan, karena dataset mentah biasanya terdapat missing data dan noise sehingga data berikutnya menjadi lengkap dan konsisten.

Untuk membuat kode dalam python dibutuhkan beberapa library tambahan yaitu.

* [skelarn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)

Cara menginstall library dengan command prompt:

pastikan direktori sudah pada tempat dimana python diinstall, sebagai contoh C:\Python27>
```
pip install numpy
```

kemudian scipy

```
pip install scipy
```

install sklearn

```
pip install -U scikit-learn
```


Penggunaan pada kode python
========================================

```python
from importlib import import_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Memasukan dataset pada variabel

Python dapat membaca file dataset csv dengan bantuan pandas.

```python
dataset = pd.read_csv('Nama-File.csv')
```

### Memilih kolom dan baris

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -3].values
```

variabel X dan y telah menampung nilai dari semua baris dan X semua kolom selain kolom index 1 dari belakang dan y kolom index ke 3 dari belakang. Jika kurang jelas bisa dilihat refrensi dari [refrensi](https://saltfarmer.github.io/blog/machine%20learning/Belajar-Machine-Learning-pandas/).

### Melakukan split data menjadi Training set dan Testing set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Training data akan digunakan untuk mencoba suatu algoritma, dan testing akan digunakan untuk mengetahui performa algoritma yang dilatih sebelumnya untuk menemukan data baru atau yang belum pernah dilihat sebelumnya.

Berikutnya menangani nilai kosong pada dataset atau bernilai NaN

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X)
X = imputer.transform(X)
```

Pada kode diatas missing_values=np.nan yang berarti target data yang akan dianggap missing adalah data yang memiliki nilai NaN atau kosong, dengan metode most_frequent atau yang paling banyak muncul.

[SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) terdapat berbagai strategy antara lain mean, median, most_frequent dan constant.

## Encoding data kategori

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 3, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
```

[ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) memiliki beberapa parameter yang harus diatur yaitu nama transformer OneHotEncode(), dan index kolom dari [dataset](https://github.com/n0tavaliduser/Data-Mining/blob/main/Assignment-1/data-jumlah-armada-bus-sekolah-2017.csv) sebagai contoh yang harus diencoding adalah index ke 0, 1, 3 dan 5 lalu remainder untuk mengatur kolom sisanya akan disertakan dalam output 'passthrough' atau tidak 'drop'.

### Melakukan split data menjadi Training set dan Testing set

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("X_train : ", X_train)
print("X_test : ", X_test)
print("y_train : ", y_train)
print("y_test : ", y_test)
```

Melakukan feature scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:] = sc.fit_transform(X_train[:, 5:].reshape(1, -1))
X_test[:, 5:] = sc.fit_transform(X_test[:, 5:].reshape(1, -1))
print(X_test)
print(X_train)
```
