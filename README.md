# COMP9417_GroupProject
This project has three folders:

(1) codes

·preprocess.py (API for reading datasets and doing some preprocessing)

preprocess_dataset.py (Implement preprocess.py API, this program should run first to produce training and test set for following training, tune and prediction: preprocessed_train.feather, preprocessed_test.feather)

skmem.py (API from a Kaggle competitor for memory reduction and optimization of dataset, source: https://www.kaggle.com/jpmiller/skmem)

random_forests.py (main program for training, tuning and prediction with random forests model)

random_forests_parameters_plot.py (After training, plotting graphs helps tuning each parameters, the plotted graphs are placed in "plot" folder)

ridge.py (including the steps of optimize the hyperparameter and ploting the features weight graph. The plots are also placed in "plot" folder)

xgb.py (this is only for testing the performance of xgboost in the train set with relatively high n_estimator for the optimization spending too much time)

(2) data

preprocessed_train.feather (preprocessed training set, can be implemented directly by model to perform 5-fold cross-validation and training)

preprocessed_test.feather (preprocessed test set, can be implemented directly by model to predict final result that submitted to Kaggle)

train.csv (674MB)

test.csv (1.36GB)

weather_train.csv (7.1MB)

weather_test.csv (14.1MB)

building_metadata.csv (44.4KB)

Above five datasets are not exist originally since they are relatively big, we cannot put them in the submission file, but can be downloaded on the source: https://www.kaggle.com/c/ashrae-energy-prediction/data

******
Note that If preprocess.py and preprocess_dataset.py are required to test, these files must be downloaded and place in the "data" folder, if no need, just ignore these two programs and start to run main program of each models.
******

(3) plot

there are two sub-folder contains graphs of each model. These graphs can also be seen in report. 

xgb.py will not generate any plot but the performance report in command line.
