# -*- coding: UTF-8 -*-
import sys
import numpy as np
import datetime

from sklearn.model_selection import GridSearchCV

from feature_word2vec import getvalue_train
from feature_word2vec import getvalue_valid
from sklearn.ensemble import RandomForestClassifier


sys.path.append('./')

start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

ids = np.load('idsMatrix.npy')
y = getvalue_train()
ids_valid = np.load('idsMatrix_valid.npy')
y_valid = getvalue_valid()
X = np.concatenate((ids, ids_valid))
y = np.concatenate((y, y_valid))

rf0 = RandomForestClassifier(oob_score=True)
rf0.fit(X, y)
print rf0.score(X, y)
print rf0.oob_score_

param_test1 = {'n_estimators': range(10, 50, 10), 'max_depth': range(15, 30, 2), 'min_samples_split': range(2, 10, 2)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=param_test1, scoring='accuracy', cv=5)
gsearch1.fit(X, y)
print gsearch1.cv_results_
print gsearch1.best_params_
print gsearch1.best_score_

param_test2 = {'max_features': range(50, 60, 2)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=40, min_samples_split=6, max_depth=27),
                        param_grid=param_test2, scoring='accuracy', cv=5)
gsearch2.fit(X, y)
print gsearch2.cv_results_
print gsearch2.best_params_
print gsearch2.best_score_

rf1 = RandomForestClassifier(n_estimators=40, min_samples_split=6, max_depth=27, max_features=56, oob_score=True)
rf1.fit(X, y)
print rf1.score(X, y)
print rf1.oob_score_