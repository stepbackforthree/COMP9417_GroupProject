import math
import numpy as np
import pandas as pd
import pyarrow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pandas.core.frame import DataFrame

# this program is the main program of Random Forests Regression including tuning and prediction

# read dataset from two preprocessed feather files
train_set = pd.read_feather('../data/preprocessed_train.feather')
test_set = pd.read_feather('../data/preprocessed_test.feather')

# features set
y = train_set['meter_reading']
# target set
X = train_set.drop(columns='meter_reading')


# find best parameter n_estimators
def n_estimators_tune():
    # using 5-fold cross-validation to tune parameter
    kf = KFold(n_splits=5, shuffle=False)
    iteration = 1
    validation_mean_error_collection = []
    n_estimators_collection = range(10, 151, 10)
    for n_estimators in n_estimators_collection:
        validation_error_collection = []
        for train_index, val_index in kf.split(X):
            train_X, train_y = X.iloc[train_index], y[train_index]
            val_X, val_y = X.iloc[val_index], y[val_index]
            rf = RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_samples=300000, max_features=0.5,
                                       min_samples_leaf=3)
            rf.fit(train_X, train_y)
            predict_y = rf.predict(val_X)
            print(
                f'iteration {iteration}: single validation error: {np.sqrt(mean_squared_error(predict_y, val_y))}')
            print(rf)
            iteration = iteration + 1
            validation_error_collection.append(np.sqrt(mean_squared_error(predict_y, val_y)))
        validation_mean_error_collection.append(np.mean(validation_error_collection))
    print('validation RMSE:', validation_mean_error_collection)
    print('average validation RMSE:', np.mean(validation_mean_error_collection))
    print('best n_estimators:', n_estimators_collection[validation_mean_error_collection.index(min(validation_mean_error_collection))])


# get n_estimators = 110
# find best parameter max_depth
def max_depth_tune():
    kf = KFold(n_splits=5, shuffle=False)
    iteration = 1
    validation_mean_error_collection = []
    max_depth_collection = range(1, 50, 2)
    for max_depth in max_depth_collection:
        validation_error_collection = []
        for train_index, val_index in kf.split(X):
            train_X, train_y = X.iloc[train_index], y[train_index]
            val_X, val_y = X.iloc[val_index], y[val_index]
            if max_depth == 1:
                rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=300000, max_features=0.5,
                                           min_samples_leaf=3, max_depth=None)
            else:
                rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=300000, max_features=0.5,
                                           min_samples_leaf=3, max_depth=max_depth)
            rf.fit(train_X, train_y)
            predict_y = rf.predict(val_X)
            print(
                f'iteration {iteration}: single validation error: {np.sqrt(mean_squared_error(predict_y, val_y))}')
            print(rf)
            iteration = iteration + 1
            validation_error_collection.append(np.sqrt(mean_squared_error(predict_y, val_y)))
        validation_mean_error_collection.append(np.mean(validation_error_collection))
    print('validation RMSE:', validation_mean_error_collection)
    print('average validation RMSE:', np.mean(validation_mean_error_collection))
    print('best n_estimators:', max_depth_collection[validation_mean_error_collection.index(min(validation_mean_error_collection))])


# get max_depth = 47
# find best parameter min_sample_leaf
def min_sample_leaf_tune():
    kf = KFold(n_splits=5, shuffle=False)
    iteration = 1
    validation_mean_error_collection = []
    min_sample_leaf_collection = range(1, 10, 1)
    for min_sample_leaf in min_sample_leaf_collection:
        validation_error_collection = []
        for train_index, val_index in kf.split(X):
            train_X, train_y = X.iloc[train_index], y[train_index]
            val_X, val_y = X.iloc[val_index], y[val_index]
            rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=300000, max_features=0.5,
                                       min_samples_leaf=min_sample_leaf, max_depth=47)
            rf.fit(train_X, train_y)
            predict_y = rf.predict(val_X)
            print(
                f'iteration {iteration}: single validation error: {np.sqrt(mean_squared_error(predict_y, val_y))}')
            print(rf)
            iteration = iteration + 1
            validation_error_collection.append(np.sqrt(mean_squared_error(predict_y, val_y)))
        validation_mean_error_collection.append(np.mean(validation_error_collection))
    print('validation RMSE:', validation_mean_error_collection)
    print('average validation RMSE:', np.mean(validation_mean_error_collection))
    print('best n_estimators:', min_sample_leaf_collection[validation_mean_error_collection.index(min(validation_mean_error_collection))])


# get min_sample_leaf = 2
# find best parameter max_feature
def max_feature_tune():
    kf = KFold(n_splits=5, shuffle=False)
    iteration = 1
    validation_mean_error_collection = []
    max_feature_collection = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'auto', 'sqrt', 'log2']
    for max_feature in max_feature_collection:
        validation_error_collection = []
        for train_index, val_index in kf.split(X):
            train_X, train_y = X.iloc[train_index], y[train_index]
            val_X, val_y = X.iloc[val_index], y[val_index]
            rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=300000, max_features=max_feature,
                                       min_samples_leaf=2, max_depth=47)
            rf.fit(train_X, train_y)
            predict_y = rf.predict(val_X)
            print(
                f'iteration {iteration}: single validation error: {np.sqrt(mean_squared_error(predict_y, val_y))}')
            print(rf)
            iteration = iteration + 1
            validation_error_collection.append(np.sqrt(mean_squared_error(predict_y, val_y)))
        validation_mean_error_collection.append(np.mean(validation_error_collection))
    print('validation RMSE:', validation_mean_error_collection)
    print('average validation RMSE:', np.mean(validation_mean_error_collection))
    print('best n_estimators:', max_feature_collection[validation_mean_error_collection.index(min(validation_mean_error_collection))])


# get max_feature = 0.7
# find best parameter max_sample
def max_sample_tune():
    kf = KFold(n_splits=5, shuffle=False)
    iteration = 1
    validation_mean_error_collection = []
    max_sample_collection = range(50000, 1000001, 50000)
    for max_sample in max_sample_collection:
        validation_error_collection = []
        for train_index, val_index in kf.split(X):
            train_X, train_y = X.iloc[train_index], y[train_index]
            val_X, val_y = X.iloc[val_index], y[val_index]
            rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=max_sample, max_features=0.7,
                                       min_samples_leaf=2, max_depth=47)
            rf.fit(train_X, train_y)
            predict_y = rf.predict(val_X)
            print(
                f'iteration {iteration}: single validation error: {np.sqrt(mean_squared_error(predict_y, val_y))}')
            print(rf)
            iteration = iteration + 1
            validation_error_collection.append(np.sqrt(mean_squared_error(predict_y, val_y)))
        validation_mean_error_collection.append(np.mean(validation_error_collection))
    print('validation RMSE:', validation_mean_error_collection)
    print('average validation RMSE:', np.mean(validation_mean_error_collection))
    print('best n_estimators:', max_sample_collection[validation_mean_error_collection.index(min(validation_mean_error_collection))])
    

# get the final tuned model and implement 5-fold cross-validation once again
def final_model_validation():
    kf = KFold(n_splits=5, shuffle=False)
    iteration = 1
    validation_error_collection = []
    for train_index, val_index in kf.split(X):
        train_X, train_y = X.iloc[train_index], y.iloc[train_index]
        val_X, val_y = X.iloc[val_index], y.iloc[val_index]
        rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=600000, max_features=0.7,
                                   min_samples_leaf=2, max_depth=47)
        rf.fit(train_X, train_y)
        predict_validation_y = rf.predict(val_X)
        print(
            f'iteration {iteration}: single validation error: {np.sqrt(mean_squared_error(predict_validation_y, val_y))}')
        print(rf)
        iteration = iteration + 1
        validation_error_collection.append(np.sqrt(mean_squared_error(predict_validation_y, val_y)))
    print('validation RMSE:', validation_error_collection)
    print('average validation RMSE:', np.mean(validation_error_collection))


# using final tuned model to predict the final result for Kaggle based on test dataset
def final_model_predict():
    row_id_list = test_set['row_id']
    test_X = test_set.drop(columns='row_id')
    rf = RandomForestRegressor(n_jobs=-1, n_estimators=110, max_samples=600000, max_features=0.7,
                               min_samples_leaf=2,max_depth=47)
    print(rf)
    rf.fit(X, y)
    log_predict_test_y = rf.predict(test_X)
    predict_test_y = math.e ** log_predict_test_y - 1
    submission = DataFrame({'row_id': row_id_list, 'meter_reading': predict_test_y})
    # WARNING: this command will produce a 1.08G file and take several minutes
    submission.to_csv('../data/submission.csv', index=False)


# main entrance of program
# WARNING: these commands will call random forests regressor and process 20 million samples
# it would make all CPU core work, require beyond 20G memory space and take long time (several hours) to complete
# please feel free to comment out each commands to see single tuning performance

# n_estimators_tune()
# max_depth_tune()
# min_sample_leaf_tune()
# max_feature_tune()
# max_sample_tune()
# final_model_validation()
# final_model_predict()

