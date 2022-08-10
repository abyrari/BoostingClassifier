import numpy as np
from skimage import feature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC


X=feature_all
class1=np.ones((106,1))
class2=np.zeros((496,1))
Y=np.concatenate((class1,class2),axis=0)

kf = KFold(n_splits=5)
kf.get_n_splits(X)
print(kf)
KFold(n_splits=5, random_state=10, shuffle=True)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X,Y)