# The range of Alpha here is only the final range because it will spend a long time to get this range. The step before
# could be find in the report.
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

train_set = pd.read_feather('../data/preprocessed_train.feather')
print(train_set.shape)
y = train_set['meter_reading']
X = train_set.drop(columns='meter_reading')
print(X.shape)
print(X)
print(type(y))
print(type(y[0]))

kf = KFold(n_splits=5, shuffle=False)

'''
alpha = 10000000
max_alpha = 50000000
while step >= 1
    step = (max_alpha - alpha) / 4
    error_box = []
    alpha_box = []
    
    while alpha <= max_alpha:
        iteration = 1
        sum_error = 0
        rf = Ridge(alpha=alpha)
        for train_index, test_index in kf.split(X):
            train_X, train_y = X.iloc[train_index], y[train_index]
            test_X, test_y = X.iloc[test_index], y[test_index]
            rf.fit(train_X, train_y)
            predict_train_y = rf.predict(train_X)
            predict_test_y = rf.predict(test_X)
            test_error = np.sqrt(mean_squared_error(predict_test_y, test_y))
            print(f'iteration {iteration}: train error: {np.sqrt(mean_squared_error(predict_train_y, train_y))}   test error: {test_error}')
            sum_error = sum_error + test_error
            iteration += 1
        error_box.append(sum_error)
        alpha_box.append(alpha)
        print("Alpha = " + str(alpha) + ": " + str(sum_error))
        alpha = alpha + step
    alpha = alpha_box[error_box.index(min(error_box)) - 1]
    max_alpha = alpha_box[error_box.index(min(error_box)) + 1]
'''

# Code above is the completely step of optimization but will spend a lot of time, at least in my device
# Code below is the final range, which is same as or very similar to code above after 10+ iteration

start = time.time()
alpha = 14508806.89375
max_alpha = 14508807.60625
step = (max_alpha - alpha) / 4
error_box = []
alpha_box = []

while alpha <= max_alpha:
    iteration = 1
    sum_error = 0
    rf = Ridge(alpha=alpha)
    for train_index, test_index in kf.split(X):
        train_X, train_y = X.iloc[train_index], y[train_index]
        test_X, test_y = X.iloc[test_index], y[test_index]
        rf.fit(train_X, train_y)
        predict_train_y = rf.predict(train_X)
        predict_test_y = rf.predict(test_X)
        test_error = np.sqrt(mean_squared_error(predict_test_y, test_y))
        print(f'iteration {iteration}: train error: {np.sqrt(mean_squared_error(predict_train_y, train_y))}   test error: {test_error}')
        sum_error = sum_error + test_error
        iteration += 1
    error_box.append(sum_error)
    alpha_box.append(alpha)
    print("Alpha = " + str(alpha) + ": " + str(sum_error))
    alpha = alpha + step

end = time.time()
print("Cost time: " + str(end-start))
print(alpha_box)
print(error_box)
print(step)
print(error_box.index(min(error_box)))
plt.plot(alpha_box, error_box)
plt.xlabel("Alpha")
plt.ylabel("Sum_Error")
plt.scatter(alpha_box[error_box.index(min(error_box))], min(error_box))
plt.savefig("../plot/ridge/Ridge_final_range.png")
plt.clf()

rf = Ridge(alpha=alpha_box[error_box.index(min(error_box))])
rf.fit(X, y)
coe = list(abs(rf.coef_))
print(coe)
sum_coe = sum(coe)
for i in range(len(coe)):
    coe[i] = coe[i] / sum_coe
print(coe)
label = X.columns
for i in range(len(label)):
    plt.barh(label[i], coe[i])
plt.xlabel("feature importance")
plt.savefig("../plot/ridge/Ridge_features_importance.png")
