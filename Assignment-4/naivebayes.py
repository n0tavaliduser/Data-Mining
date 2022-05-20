from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

class naivebayes(object):

    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sc = StandardScaler()
        self.classifier = GaussianNB()
        self.fittedClassifier = None
        self.y_pred = None
        self.x_set = None
        self.y_set = None
        self.x1 = None
        self.x2 = None

    def setDataset(self, namaFile):
        self.dataset = pd.read_csv(namaFile)

    def getDataset(self):
        return(self.dataset)

    def setX(self, how):
        index = []
        for i in range(how):
            index.append(int(input("index {} -> ".format(i+1))))
        self.X = self.dataset.iloc[:, index].values
    
    def getX(self):
        return(self.X)

    def setY(self, index):
        self.y = self.dataset.iloc[:, index]

    def setTrainTestSplit(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)

    def getXtrain(self):
        return(self.X_train)

    def getXtest(self):
        return(self.X_test)

    def getYtrain(self):
        return(self.y_train)
                                                                                                                                                                                                                                               
    def getYtest(self):
        return(self.y_test)

    def getClassifier(self):
        return(self.classifier)

    def show(self):
        self.classifier.fit(self.X_train, self.y_train)
        plt.contourf(self.x1, self.x2, self.classifier.predict(np.array([self.x1.ravel(), self.x2.ravel()]).T).reshape(self.x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(self.x1.min(), self.x1.max())
        plt.xlim(self.x2.min(), self.x2.max())
        for i, j in enumerate (np.unique(self.y_set)):
            plt.scatter(self.x_set[self.y_set == j, 0], self.x_set[self.y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Klasifikasi Data dengan Naive Bayes (Data Training)')
        plt.xlabel('umur')
        plt.ylabel('Estimasi Gaji')
        plt.legend()
        plt.show()

    def setXtrainXtestSscaler(self):
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.fit_transform(self.X_test)

    def setX_setY_set(self):
        self.x_set, self.y_set = self.X_train, self.y_train

    def setX1X2(self):
        self.x1, self.x2 = np.meshgrid(np.arange(start = self.x_set[:, 0].min()-1, stop = self.x_set[:, 0].max() + 1, step = 0.01), np.arange(start = self.x_set[:, -1].min()-1, stop = self.x_set[:, 0].max() + 1, step = 0.01))

    def startClassification(self):
        self.setTrainTestSplit()
        self.setXtrainXtestSscaler()
        self.setX_setY_set()
        self.setX1X2()
        self.show()

if __name__ == "__main__":
    nb = naivebayes()
    nb.setDataset('databus.csv')
    nb.setX(2)
    nb.setY(2)