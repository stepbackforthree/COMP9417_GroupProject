import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# this program is the plot program helping parameters tuning
# real value lists generated from random_forests.py are used for easier plot


def n_estimators():
    n_estimators_collection = range(10, 151, 10)
    n_estimators_error_collection = [1.111474709057307, 1.099385309033329, 1.0909948609984013, 1.0893069708838865,
                                     1.087652771680973, 1.082395365768208, 1.0814237365741657, 1.084450239924885,
                                     1.0841991883917736, 1.0818961231199533, 1.0804656796298404, 1.080609283532565,
                                     1.0812506374066477, 1.0809049468830334, 1.0826997516804986]
    min_y = min(n_estimators_error_collection)
    min_x = n_estimators_collection[n_estimators_error_collection.index(min(n_estimators_error_collection))]
    plt.plot(n_estimators_collection, n_estimators_error_collection)
    plt.scatter(min_x, min_y, color='red')
    plt.text(min_x, min_y + 0.003, f"({min_x}, {round(min_y, 6)})")
    plt.xlabel('n_estimators')
    plt.ylabel('average of root mean squared error')
    plt.title('average root mean squared error changes with the number of trees', fontsize=10)
    plt.savefig('../plot/random_forests/n_estimators.png')


def max_depth():
    max_depth_collection = list(range(1, 50, 2))
    max_depth_error_collection = [1.08235529091921, 1.902224401546507, 1.7887730960632624, 1.6512961797155772,
                                  1.5018875821618534, 1.3622287510329218, 1.2423422940960862, 1.1664626408415715,
                                  1.120677300188432, 1.0963289237599163, 1.0899901892264832, 1.0876149664324601,
                                  1.0836089948052054, 1.0814438555235566, 1.0822600661547233, 1.0817158904884345,
                                  1.0825176864924084, 1.08103878587075, 1.0810179252455612, 1.0817493117833707,
                                  1.0819635919447461, 1.0855318018490796, 1.0800616036927315, 1.0793851105196446,
                                  1.080734971778035]

    min_y = min(max_depth_error_collection)
    min_x = max_depth_collection[max_depth_error_collection.index(min(max_depth_error_collection))]
    max_y = max(max_depth_error_collection)
    max_x = max_depth_collection[max_depth_error_collection.index(max(max_depth_error_collection))]
    plt.plot(max_depth_collection, max_depth_error_collection, color='purple')
    plt.scatter(min_x, min_y, color='red')
    plt.text(min_x, min_y + 0.05, f"({min_x}, {round(min_y, 6)})")
    plt.text(max_x + 2, max_y, f"({max_x}, {round(max_y, 6)})")
    plt.xlabel('max_depth')
    plt.ylabel('average of root mean squared error')
    plt.title('average root mean squared error changes with the maximum depth of trees', fontsize=10)
    plt.savefig('../plot/random_forests/max_depth.png')


def min_sample_leaf():
    min_sample_leaf_collection = range(1, 10, 1)
    min_sample_leaf_error_collection = [1.0774624122221121, 1.0715973994409604, 1.0775890703183402, 1.079208962031292,
                                        1.0792823091231207, 1.0860430264326753, 1.0906934929537475, 1.091497028982917,
                                        1.1000589609048836]

    min_y = min(min_sample_leaf_error_collection)
    min_x = min_sample_leaf_collection[min_sample_leaf_error_collection.index(min(min_sample_leaf_error_collection))]
    plt.plot(min_sample_leaf_collection, min_sample_leaf_error_collection, color='green')
    plt.scatter(min_x, min_y, color='red')
    plt.text(min_x + 0.2, min_y, f'({min_x}, {round(min_y, 6)})')
    plt.xlabel('min_sample_leaf')
    plt.ylabel('average of root mean squared error')
    plt.title('average root mean squared error changes with the minimum samples of a leaf', fontsize=10)
    plt.savefig('../plot/random_forests/min_sample_leaf.png')


def max_features():
    max_features_collection = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
    max_features_collection_label = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'auto', 'sqrt', 'log2']
    max_feature_error_collection = [1.5964964082159068, 1.2257138475197245, 1.1440863708306368, 1.0813310977289585,
                                    1.0737490491310755, 1.0649760550373304, 1.0623693990708905, 1.0689479244828035,
                                    1.068175502894198, 1.076117257899425, 1.2246273266870689, 1.2213555506404612]

    min_y = min(max_feature_error_collection)
    min_x = max_features_collection[max_feature_error_collection.index(min(max_feature_error_collection))]
    plt.plot(max_features_collection, max_feature_error_collection, color='orange')
    plt.scatter(min_x, min_y, color='red')
    plt.xticks(max_features_collection, max_features_collection_label)
    plt.text(min_x, min_y + 0.03, f'({min_x}, {round(min_y, 6)})')
    plt.xlabel('max_features')
    plt.ylabel('average of root mean squared error')
    plt.title('average root mean squared error changes with the maximum number of features', fontsize=10)
    plt.savefig('../plot/random_forests/max_features.png')


def max_samples():
    max_samples_collection = range(50000, 750001, 50000)
    max_samples_error_collection = [1.1313875402028806, 1.0899236759063502, 1.0725907486767947, 1.0674295699300411,
                                    1.060342666559071, 1.0593808513292597, 1.0559969941773075, 1.0564180633169358,
                                    1.0542674800427072, 1.0553405909585507, 1.0559113878029016, 1.0532907736624486,
                                    1.0541188692345478, 1.054940885514087, 1.0557533314108487]

    min_y = min(max_samples_error_collection)
    min_x = max_samples_collection[max_samples_error_collection.index(min_y)]
    plt.plot(max_samples_collection, max_samples_error_collection, color='pink')
    plt.scatter(min_x, min_y, color='red')
    plt.text(min_x - 50000, min_y + 0.005, f'({min_x}, {round(min_y, 6)})')
    plt.xlabel('max_samples')
    plt.ylabel('average of root mean squared error')
    plt.title('average root mean squared error changes with the number of samples', fontsize=10)
    plt.savefig('../plot/random_forests/max_samples.png')


def features_importance():
    importance_collection = [0.19733923, 0.11668446, 0.05121619, 0.0447355 , 0.31991552,
                             0.06461354, 0.00332134, 0.03599194, 0.00440643, 0.01511498,
                             0.00921555, 0.0075572 , 0.02588458, 0.01601153, 0.087992]
    feature_collection = ['building_id',
                          'meter',
                          'site_id',
                          'primary_use',
                          'square_feet',
                          'air_temperature',
                          'cloud_coverage',
                          'dew_temperature',
                          'precip_depth_1_hr',
                          'sea_level_pressure',
                          'wind_direction',
                          'wind_speed',
                          'hour',
                          'day',
                          'week']

    features_importance_dict = {}
    for i in range(len(feature_collection)):
        features_importance_dict[feature_collection[i]] = importance_collection[i]

    ordered_dict = sorted(features_importance_dict.items(), key=lambda x:x[1], reverse=True)

    feature_collection = []
    importance_collection = []
    for i in range(len(ordered_dict)):
        feature_collection.append(ordered_dict[i][0])
        importance_collection.append(ordered_dict[i][1])

    cmap = cm.get_cmap('jet', len(ordered_dict))
    colors = cmap(np.linspace(0, 1, len(ordered_dict)))
    plt.figure(figsize=[11.5, 7], dpi=100)
    plt.barh(feature_collection, importance_collection, color=colors)
    plt.xlabel('feature importance')
    plt.title('Gini importance of each features', fontsize=10)
    plt.savefig('../plot/random_forests/features_importance.png')


# main entrance of plot program
n_estimators()
plt.clf()
max_depth()
plt.clf()
min_sample_leaf()
plt.clf()
max_features()
plt.clf()
max_samples()
plt.clf()
features_importance()