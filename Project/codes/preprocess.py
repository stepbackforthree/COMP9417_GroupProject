import numpy as np
import pandas as pd
import skmem

# this program will read five original datasets and preprocess them


class Preprocess:

    def __init__(self):
        # read five original datasets
        self.train_set = pd.read_csv('../data/train.csv')
        self.test_set = pd.read_csv('../data/test.csv')
        self.weather_train_set = pd.read_csv('../data/weather_train.csv')
        self.weather_test_set = pd.read_csv('../data/weather_test.csv')
        self.building_metadata = pd.read_csv('../data/building_metadata.csv')

    def data_merge(self):
        # combine each datasets to produce original training and test set
        self.train_set = pd.merge(self.train_set, self.building_metadata, on='building_id', how='left')
        self.train_set = pd.merge(self.train_set, self.weather_train_set, on=['site_id', 'timestamp'], how='left')
        # implement API from Kaggle's competitor to decrease memory use
        # source: https://www.kaggle.com/jpmiller/skmem
        mr = skmem.MemReducer()
        self.train_set = mr.fit_transform(self.train_set)

        self.test_set = pd.merge(self.test_set, self.building_metadata, on='building_id', how='left')
        self.test_set = pd.merge(self.test_set, self.weather_test_set, on=['site_id', 'timestamp'], how='left')
        mr = skmem.MemReducer()
        self.test_set = mr.fit_transform(self.test_set)

    def categorical(self):
        # transfer string type of data to integer for easier fit
        self.train_set['primary_use'] = pd.Categorical(self.train_set['primary_use']).codes
        self.test_set['primary_use'] = pd.Categorical(self.test_set['primary_use']).codes

    def drop_data(self):
        # drop some features that contains many missing value
        self.train_set.drop(columns=['year_built', 'floor_count'], inplace=True)
        self.test_set.drop(columns=['year_built', 'floor_count'], inplace=True)

        # transfer timestamp type to individual integer "hour", "day" and "week" for easier fit
        self.train_set['timestamp'] = pd.to_datetime(self.train_set['timestamp'])
        self.test_set['timestamp'] = pd.to_datetime(self.test_set['timestamp'])

        self.train_set['hour'] = self.train_set['timestamp'].dt.hour
        self.train_set['day'] = self.train_set['timestamp'].dt.day
        self.train_set['week'] = self.train_set['timestamp'].dt.isocalendar().week
        self.train_set.drop(columns='timestamp', inplace=True)

        self.test_set['hour'] = self.test_set['timestamp'].dt.hour
        self.test_set['day'] = self.test_set['timestamp'].dt.day
        self.test_set['week'] = self.test_set['timestamp'].dt.isocalendar().week
        self.test_set.drop(columns='timestamp', inplace=True)

    def fill_na(self):
        # fill missing value with mean of each feature
        na_features = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                       'sea_level_pressure', 'wind_direction', 'wind_speed']
        for feature in na_features:
            self.train_set[feature] = self.train_set[feature].fillna(self.train_set[feature].mean())
            self.test_set[feature] = self.test_set[feature].fillna(self.test_set[feature].mean())

    def target_smooth(self):
        # smooth the target values for easier expression of result
        self.train_set['meter_reading'] = np.log1p(self.train_set['meter_reading'])

    def process(self):
        self.data_merge()
        self.categorical()
        self.drop_data()
        self.fill_na()
        self.target_smooth()
        # store the final preprocessed training and test set to feather files
        # for lower file size and easier implementation
        self.train_set.to_feather('../data/preprocessed_train.feather')
        self.test_set.to_feather('../data/preprocessed_test.feather')

