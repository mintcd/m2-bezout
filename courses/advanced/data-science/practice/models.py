import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# 1. Load your MNIST data once
X_train = np.load('data/mnist1_features_train.npy', allow_pickle=True)
y_train = np.load('data/mnist1_labels_train.npy', allow_pickle=True)

# 2. Identify which model to run from the command line argument
model_type = sys.argv[1]

if model_type == "knn":
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 2, 3, 4, 5]}
    clf = GridSearchCV(knn, parameters, cv=3)

elif model_type == "svc":
    svc = LinearSVC(max_iter=5000)
    parameters = {'C': np.logspace(-8, 8, 17, base=2)}
    clf = GridSearchCV(svc, parameters, cv=3)

elif model_type == "svc_maxabs":
    svc = LinearSVC(max_iter=5000)
    pipe = Pipeline([('scaler', MaxAbsScaler()), ('svc', svc)])
    parameters = {'svc__C': np.logspace(-8, 8, 17, base=2)}
    clf = GridSearchCV(pipe, parameters, cv=3)

elif model_type == "logreg_std":
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=5000))])
    parameters = {'logreg__C': np.logspace(-8, 8, 17, base=2)}
    clf = GridSearchCV(pipe, parameters, cv=3)

else:
    sys.exit("Unknown model type")

# 3. Train the model (This is where the heavy FLOPs occur)
clf.fit(X_train, y_train)