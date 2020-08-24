import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#ff000','#00ff00','#0000ff'])

iris = datasets.load_iris()
X,y = iris.data, iris.target

"""
X_train:training samples
X_test: test samples
y_train: training lables
y_test : test lables
"""
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state =1234)


# shape of input--rows & columns - samples & lables
print(X_train.shape)
print(X_train[0])#features of first row

# shape of output-only one colum
print(y_train.shape)
print(y_train)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y, cmap = cmap, edgecolor ='k',s=20)
# plt.show()

#  test sample
# a = [1,1,1,2,2,3,4,5,6]
# from collections import Counter
# most_common = Counter(a).most_common(1)
# print(most_common[0][0])

# importing KNN module
from knn import KNN

# classifier clf
clf = KNN(k=5)

# fit method training data
clf.fit(X_train,y_train)

#  predict test sample
predictions = clf.predict(X_test)

# test accuracy
accuracy = np.sum(predictions == y_test)/len(y_test)
print(accuracy)