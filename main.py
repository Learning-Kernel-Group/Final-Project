import numpy as np
from matplotlib import pyplot as plt
import train_test as tes
import plot


class Problem():

    def __init__(self, dataset_name, method_name, folds, lamb_range, eta, norm_bound, tolerence, subsampling):
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.lamb_range = lamb_range
        self.folds = folds
        self.eta = eta
        self.norm_bound = norm_bound
        self.tolerence = tolerence
        self.subsampling = subsampling
        self.error_array = []
        self.error_arr_array = []
        self.training_result = None
        self.quad_ker = None

        self.features, self.labels, self.testing_features, self.testing_labels = self.import_data_from()
        self.n, self.p = self.features.shape
        self.mu_0 = np.zeros(self.p)

    def import_data_from(self):
        if self.dataset_name == 'ionosphere':
            features = np.load(
                './Data Sets/UCI Data Sets/ionosphere_features_train.npy')
            labels = np.load(
                './Data Sets/UCI Data Sets/ionosphere_labels_train.npy')
            testing_features = np.load(
                './Data Sets/UCI Data Sets/ionosphere_features_test.npy')
            testing_labels = np.load(
                './Data Sets/UCI Data Sets/ionosphere_labels_test.npy')
        if self.dataset_name == 'kin8nm':
            features = np.load(
                './Data Sets/regression-datasets-kin8nm_features_train.npy')
            labels = np.load(
                './Data Sets/regression-datasets-kin8nm_labels_train.npy')
            testing_features = np.load(
                './Data Sets/regression-datasets-kin8nm_features_test.npy')
            testing_labels = np.load(
                './Data Sets/regression-datasets-kin8nm_labels_test.npy')
        if self.dataset_name == 'sonar':
            features = np.load(
                './Data Sets/UCI Data Sets/sonar_features_train.npy')
            labels = np.load(
                './Data Sets/UCI Data Sets/sonar_labels_train.npy')
            testing_features = np.load(
                './Data Sets/UCI Data Sets/sonar_features_test.npy')
            testing_labels = np.load(
                './Data Sets/UCI Data Sets/sonar_labels_test.npy')
        return features, labels, testing_features, testing_labels

    def cross_validation(self):
        for lamb in self.lamb_range:
            error_arr = tes.cross_validation(self.features, self.labels, self.folds, self.method_name, lamb,
                                             self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling)
            self.error_arr_array.append(error_arr)
        self.error_arr_array = np.array(self.error_arr_array)

    def train_test(self):
        for lamb in self.lamb_range:
            self.training_result, self.quad_ker = tes.training(
                self.features, self.labels, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling, method=self.method_name)
            error = tes.testing(self.training_result, self.quad_ker, self.features, self.labels, self.testing_features,
                                self.testing_labels, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling)
            self.error_array.append(error)
        self.error_array = np.array(self.error_array)

    def plotting(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot.plot_as_seq(self.error_array, self.lamb_range, ax)
        plot.plot_as_errorbar(self.error_arr_array, self.lamb_range, ax)
        plt.savefig('figure.png', dpi=200)


if __name__ == '__main__':
    # features, labels, testing_features, testing_labels = import_data_from(
    #     'ionosphere')
    # n, p = features.shape
    # error_array = []
    # error_arr_array = []
    # lamb_range = range(10, 101, 10)
    # for lamb in lamb_range:
    #     training_result, quad_ker = tes.training(
    #         features, labels, 10, 1, 1, 0.01, np.zeros(p), 1, method='pgd')
    #     error = tes.testing(training_result, quad_ker, features, labels, testing_features,
    #                         testing_labels, 10, 1, 1, 0.01, np.zeros(p), 1)
    #     error_arr = tes.cross_validation(
    #         features, labels, 10, 10, 1, 1, 0.01, np.zeros(p), 1)
    #     error_array.append(error)
    #     error_arr_array.append(error_arr)
    # error_array = np.array(error_array)
    # error_arr_array = np.array(error_arr_array)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plot.plot_as_seq(error_array, lamb_range, ax)
    # plot.plot_as_errorbar(error_arr_array, lamb_range, ax)
    # plt.savefig('figure.png', dpi=200)

    lamb_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    problem = Problem('ionosphere', 'pgd', 10, lamb_range, 1, 1, 0.01, 1)
    problem.cross_validation()
    problem.train_test()
    problem.plotting()
