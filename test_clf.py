#!/bin/env python
import sys
from functools import partial

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from utils import load_data
from Dagger import Env

env = Env("MontezumaRevengeNoFrameskip-v0", 8)
X, Y = load_data("./played_data")
Y = [env.act_enc[y] for y in Y]


def test(model):
    global X, Y
    X = [x.flatten() for x in X]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))


def cv(model):
    global X, Y
    X = [x.flatten() for x in X]
    # sys.stdout = open("model_val.txt", "+w")
    p = partial(print, flush=True)
    p(model)
    p(cross_validate(model, X, Y, cv=5))


# test(svm.SVC(kernel="linear"))
# test(svm.SVC(kernel="sigmod"))
# test(SGDClassifier(loss="hinge"))
# test(SGDClassifier(loss="log"))
# test(DecisionTreeClassifier())
# test(MultinomialNB())
# test(RandomForestClassifier())

cv(svm.SVC(kernel="linear"))
cv(svm.SVC(kernel="sigmod"))
cv(SGDClassifier(loss="hinge"))
cv(SGDClassifier(loss="log"))
cv(DecisionTreeClassifier())
cv(MultinomialNB())
cv(RandomForestClassifier())
