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

    def process(self):
        self.data_merge()
        self.categorical()
        self.drop_data()
        return self.train_set, self.test_set

