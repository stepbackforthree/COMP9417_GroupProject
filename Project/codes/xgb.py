# This code is just for testing the performance of XGBoost. Without any optimization for the too long training time.

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

total_start = time.time()
train_set = pd.read_feather('../data/preprocessed_train.feather')
print(train_set.shape)
y = train_set['meter_reading']
X = train_set.drop(columns='meter_reading')
print(X.shape)
print(X)
print(type(y))
print(type(y[0]))
print(X.dtypes)
for i in X.columns:
    X[i] = X[i].astype(float)

kf = KFold(n_splits=5, shuffle=False)
iteration = 1
xgb = XGBRegressor()
sum_error = 0

for train_index, test_index in kf.split(X):
    start = time.time()
    train_X, train_y = X.iloc[train_index], y[train_index]
    test_X, test_y = X.iloc[test_index], y[test_index]
    xgb.fit(train_X, train_y)
    predict_train_y = xgb.predict(train_X)
    predict_test_y = xgb.predict(test_X)
    test_error = np.sqrt(mean_squared_error(predict_test_y, test_y))
    print(
        f'iteration {iteration}: train error: {np.sqrt(mean_squared_error(predict_train_y, train_y))}   test error: {test_error}')
    sum_error = sum_error + test_error
    iteration += 1
    end = time.time()
    print("Running time: " + str(end - start))

end = time.time()
print("Running time: " + str(end - total_start))

