import os
from pathlib import Path
import pickle
from sklearn.decomposition import FastICA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import logging
path_to_features = Path(os.getcwd()).parents[0]/"data/train_test_features.pkl"

logging.basicConfig(filename='training.log', level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s")
with open(path_to_features, "rb") as F:
    train_test_features = pickle.load(F)

RANDOM_STATE = 42
ica = FastICA(
    n_components=60,
    max_iter=400,
    random_state=RANDOM_STATE
)
# input_shape n_samples x n_features -> output X_train n_samples x n_components
X_train = ica.fit_transform(train_test_features["train"]["X"])
y_train = train_test_features["train"]["y"]

X_test = ica.transform(train_test_features["test"]["X"])
y_test = train_test_features["test"]["y"]

clf = SVC(random_state = RANDOM_STATE)
clf.fit(X_train)
yhat_train = clf.predict(X_train)

train_acc = round(accuracy_score(y_train, yhat_train), 2)*100

yhat_test = clf.predict(X_test)

test_acc = round(accuracy_score(y_test, yhat_test), 2)*100

logging.info(f"ICA + SVM Classifier train_acc{train_acc}, test_acc {test_acc}")