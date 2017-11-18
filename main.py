import numpy as np
from matplotlib import pyplot as plt
import testing as tes
import plot


def import_data_from(dataset_name):
    if dataset_name == 'ionosphere':
        features = np.load(
            './Data Sets/UCI Data Sets/ionosphere_features_train.npy')
        labels = np.load('./Data Sets/UCI Data Sets/ionosphere_labels_train.npy')
        testing_features = np.load(
            './Data Sets/UCI Data Sets/ionosphere_features_test.npy')
        testing_labels = np.load(
            './Data Sets/UCI Data Sets/ionosphere_labels_test.npy')
    if dataset_name == 'kin8nm':
        features = np.load(
            './Data Sets/regression-datasets-kin8nm_features_train.npy')
        labels = np.load('./Data Sets/regression-datasets-kin8nm_labels_train.npy')
        testing_features = np.load(
            './Data Sets/regression-datasets-kin8nm_features_test.npy')
        testing_labels = np.load(
            './Data Sets/regression-datasets-kin8nm_labels_test.npy')
    if dataset_name == 'sonar':
        features = np.load(
            './Data Sets/UCI Data Sets/sonar_features_train.npy')
        labels = np.load('./Data Sets/UCI Data Sets/sonar_labels_train.npy')
        testing_features = np.load(
            './Data Sets/UCI Data Sets/sonar_features_test.npy')
        testing_labels = np.load(
            './Data Sets/UCI Data Sets/sonar_labels_test.npy')
    return features, labels, testing_features, testing_labels


if __name__ == '__main__':
    features, labels, testing_features, testing_labels = import_data_from('ionosphere')
    n, p = features.shape
    error_array = []
    error_arr_array = []
    lamb_range = range(10, 101, 10)
    for lamb in lamb_range:
        error = tes.testing(features, labels, testing_features,
                        testing_labels, 10, 1, 1, 0.01, np.zeros(p), 1)
        error_arr = tes.cross_validation(
            features, labels, 10, 10, 1, 1, 0.01, np.zeros(p), 1)
        error_array.append(error)
        error_arr_array.append(error_arr)
    error_array = np.array(error_array)
    error_arr_array = np.array(error_arr_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot.plot_as_seq(error_array, lamb_range, ax)
    plot.plot_as_errorbar(error_arr_array, lamb_range, ax)
    plt.savefig('figure.png', dpi=200)
