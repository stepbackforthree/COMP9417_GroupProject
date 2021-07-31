from preprocess import Preprocess

# main entrance of preprocess API
# this program will process five original datasets and produce training and test set
# preprocessed_train.feather preprocessed_test.feather
# this program should run firstly

pr = Preprocess()
pr.process()