import pandas as pd
import skmem


class Preprocess:

    def __init__(self):
        self.train_set = pd.read_csv('../data/train.csv')
        self.test_set = pd.read_csv('../data/test.csv')
        self.weather_train_set = pd.read_csv('../data/weather_train.csv')
        self.weather_test_set = pd.read_csv('../data/weather_test.csv')
        self.building_metadata = pd.read_csv('../data/building_metadata.csv')

    def data_merge(self):
        self.train_set = pd.merge(self.train_set, self.building_metadata, on='building_id')
        self.train_set = pd.merge(self.train_set, self.weather_train_set, on=['site_id', 'timestamp'])
        mr = skmem.MemReducer()
        self.train_set = mr.fit_transform(self.train_set)

        self.test_set = pd.merge(self.test_set, self.building_metadata, on='building_id')
        self.test_set = pd.merge(self.test_set, self.weather_test_set, on=['site_id', 'timestamp'])
        mr = skmem.MemReducer()
        self.test_set = mr.fit_transform(self.test_set)

    def categorical(self):
        self.train_set['primary_use'] = pd.Categorical(self.train_set['primary_use']).codes
        self.test_set['primary_use'] = pd.Categorical(self.test_set['primary_use']).codes

    def drop_data(self):
        self.train_set.drop(columns=['year_built', 'floor_count'], inplace=True)
        self.test_set.drop(columns=['year_built', 'floor_count'], inplace=True)

        self.train_set['timestamp'] = pd.to_datetime(self.train_set['timestamp'])
        self.test_set['timestamp'] = pd.to_datetime(self.test_set['timestamp'])

        self.train_set['hour'] = self.train_set['timestamp'].dt.hour
        self.train_set['day'] = self.train_set['timestamp'].dt.day
        self.train_set['week'] = self.train_set['timestamp'].dt.isocalendar().week
        self.train_set['month'] = self.train_set['timestamp'].dt.month
        self.train_set['year'] = self.train_set['timestamp'].dt.year
        self.train_set.drop(columns='timestamp', inplace=True)

        self.test_set['hour'] = self.test_set['timestamp'].dt.hour
        self.test_set['day'] = self.test_set['timestamp'].dt.day
        self.test_set['week'] = self.test_set['timestamp'].dt.isocalendar().week
        self.test_set['month'] = self.test_set['timestamp'].dt.month
        self.test_set['year'] = self.test_set['timestamp'].dt.year
        self.test_set.drop(columns='timestamp', inplace=True)

    def fill_na(self):
        na_features = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                       'sea_level_pressure', 'wind_direction', 'wind_speed']
        for feature in na_features:
            self.train_set[feature] = self.train_set[feature].fillna(self.train_set[feature].mean())
            self.test_set[feature] = self.test_set[feature].fillna(self.test_set[feature].mean())

    def process(self):
        self.data_merge()
        self.categorical()
        self.drop_data()
        self.fill_na()
        return self.train_set, self.test_set

