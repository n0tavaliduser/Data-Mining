import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("cars.csv") 

data.head(300)

atributData = data.iloc[:,1:7].values
labelData = data.iloc[:,8].values

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean.fit(atributData[:,1:7])
atributData[:,1:7] = imp_mean.transform(atributData[:,1:7])

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(atributData, labelData, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn import tree
classifier = tree.DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_pred)

# print(classification_report(y_test, y_pred))

plt.figure(figsize=(12,12))
tree.plot_tree(classifier, feature_names=data.columns.values.tolist(), class_names=data.columns.values.tolist())
plt.title("Decision tree trained on all the cars conditions")
plt.show()