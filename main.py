import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import train_test as tes
import plot
from matplotlib import rc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


class Problem():

    def __init__(self, dataset_name, method_name, degree, folds, lamb_range, eta, norm_bound, tolerence, subsampling):
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.lamb_range = lamb_range
        self.degree = degree
        self.folds = folds
        self.eta = eta
        self.norm_bound = norm_bound
        self.tolerence = tolerence
        self.subsampling = subsampling
        self.error_array = []
        self.mse_array = []
        self.error_arr_array = []
        self.mse_arr_array = []
        self.training_result = None
        self.poly_ker = None

        self.features, self.labels, self.testing_features, self.testing_labels = self.import_data_from()
        self.n, self.p = self.features.shape
        self.mu_0 = np.zeros(self.p)

    def import_data_from(self):
        if self.dataset_name == 'kin8nm':
            features = np.load(
                './Data Sets/regression-datasets-kin8nm_features_train.npy')
            labels = np.load(
                './Data Sets/regression-datasets-kin8nm_labels_train.npy')
            testing_features = np.load(
                './Data Sets/regression-datasets-kin8nm_features_test.npy')
            testing_labels = np.load(
                './Data Sets/regression-datasets-kin8nm_labels_test.npy')
        elif self.dataset_name == 'supernova':
            features = np.load(
                './Data Sets/' + self.dataset_name + '_features_train.npy')
            labels = np.load(
                './Data Sets/' + self.dataset_name + '_labels_train.npy')
            testing_features = np.load(
                './Data Sets/' + self.dataset_name + '_features_test.npy')
            testing_labels = np.load(
                './Data Sets/' + self.dataset_name + '_labels_test.npy')
        else:
            features = np.load(
                './Data Sets/UCI Data Sets/' + self.dataset_name + '_features_train.npy')
            labels = np.load(
                './Data Sets/UCI Data Sets/' + self.dataset_name + '_labels_train.npy')
            testing_features = np.load(
                './Data Sets/UCI Data Sets/' + self.dataset_name + '_features_test.npy')
            testing_labels = np.load(
                './Data Sets/UCI Data Sets/' + self.dataset_name + '_labels_test.npy')
        return features, labels, testing_features, testing_labels

    def cross_validation(self):
        for lamb in self.lamb_range:
            error_arr, mse_arr = tes.cross_validation(self.features, self.labels, self.folds, self.method_name, self.degree, lamb,
                                                      self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling)
            self.error_arr_array.append(error_arr)
            self.mse_arr_array.append(mse_arr)
        self.error_arr_array = np.array(self.error_arr_array)
        self.mse_arr_array = np.array(self.mse_arr_array)

    def cross_validation_sk(self):
        for lamb in self.lamb_range:
            error_arr = tes.cross_validation_sk(self.features, self.labels, self.folds, self.method_name,
                                                self.degree, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling)
            self.error_arr_array.append(error_arr)
        self.error_arr_array = np.array(self.error_arr_array)

    def train_test(self):
        for lamb in self.lamb_range:
            self.training_result, self.poly_ker = tes.training(
                self.features, self.labels, self.degree, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling, method=self.method_name)
            error, mse = tes.testing(self.training_result, self.poly_ker, self.features, self.labels, self.testing_features,
                                     self.testing_labels, self.degree, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling)
            self.error_array.append(error)
            self.mse_array.append(mse)
        self.error_array = np.array(self.error_array)
        self.mse_array = np.array(self.mse_array)

    def train_test_sk(self):
        for lamb in self.lamb_range:
            svm, self.training_result, self.poly_ker = tes.training_sk(
                self.features, self.labels, self.degree, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling, method=self.method_name)

            error = tes.testing_sk(svm, self.training_result, self.poly_ker, self.features, self.labels, self.testing_features,
                                   self.testing_labels, self.degree, lamb, self.eta, self.norm_bound, self.tolerence, self.mu_0, self.subsampling)
            self.error_array.append(error)
        self.error_array = np.array(self.error_array)

    def plotting_error(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot.plot_as_seq(self.error_array, self.lamb_range, 'Test Error', ax)
        plot.plot_as_errorbar(self.error_arr_array,
                              self.lamb_range, 'Cross Validation Error', ax)
        ax.set_title(r"Data Set {} with degree $d = {{{}}}$".format(
            self.dataset_name, self.degree))
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Error rate")
        plt.legend()
        plt.savefig(
            'figure-error-{}-degree{}.png'.format(self.dataset_name, self.degree), dpi=250)
        plt.close('all')

    def plotting_mse(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot.plot_as_seq(self.mse_array, self.lamb_range, 'Test MSE', ax)
        plot.plot_as_errorbar(self.mse_arr_array,
                              self.lamb_range, 'Cross Validation MSE', ax)
        ax.set_title(r"Data Set {} with degree $d = {{{}}}$".format(
            self.dataset_name, self.degree))
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Mean Squared Error")
        plt.legend()
        plt.savefig(
            'figure-mse-{}-degree{}.png'.format(self.dataset_name, self.degree), dpi=250)
        plt.close('all')

    def benchmark_svm(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        k_range = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
        k_range = np.array(k_range)
        c_range = 2 ** k_range.astype(float)
        gamma_range = [0.5, 1, 2]

        for g in gamma_range:
            score_arr = []
            for c in c_range:
                print('Performing SVM...')
                clf = SVC(kernel='rbf', C=c, gamma=g)
                clf.fit(self.features, self.labels)
                score = clf.score(self.testing_features, self.testing_labels)
                score_arr.append(score)
            score_arr = np.array(score_arr)
            error_arr = np.ones(score_arr.shape[0]) - score_arr
            plot.plot_as_seq(error_arr, k_range,
                             r"$\gamma = {{{}}}$".format(g), ax)

        ax.set_title(
            "Data Set {} with SVM benchmark".format(self.dataset_name))
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"Classification Error")
        plt.legend()
        plt.savefig('figure-svm-{}.png'.format(self.dataset_name), dpi=250)
        plt.close('all')

    def benchmark_knn(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        k_range = range(2, 31)
        score_arr = []
        for k in k_range:
            print('Performing kNN...')
            knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
            knn.fit(self.features, self.labels)
            score = knn.score(self.testing_features, self.testing_labels)
            score_arr.append(score)
        score_arr = np.array(score_arr)
        error_arr = np.ones(score_arr.shape[0]) - score_arr
        plot.plot_as_seq(error_arr, k_range, r"kNN Classification Error", ax)

        ax.set_title(
            "Data Set {} with kNN benchmark".format(self.dataset_name))
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"Classification Error")
        plt.legend()
        plt.savefig('figure-knn-{}.png'.format(self.dataset_name), dpi=250)
        plt.close('all')


def multiprocessing_func(data_set, degree, lamb_range):
    if data_set == 'kin8nm' or data_set == 'supernova':
        problem = Problem(data_set, 'pgd', degree, 10,
                          lamb_range, 1, 1, 0.01, 100)
        problem.cross_validation_sk()
        problem.train_test_sk()
        problem.plotting_error()
    else:
        problem = Problem(data_set, 'pgd', degree,
                          10, lamb_range, 1, 1, 0.01, 1)
        problem.cross_validation_sk()
        problem.train_test_sk()
        problem.plotting_error()


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

    lamb_range = [1, 2, 4, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data_sets = ['breast-cancer', 'diabetes', 'fourclass',
                 'german', 'heart', 'ionosphere', 'sonar', 'kin8nm', 'supernova']
    processes = []
    for degree in range(1, 6):
        for data_set in data_sets:
            process = mp.Process(target=multiprocessing_func,
                                 args=(data_set, degree, lamb_range))
            processes.append(process)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # for data_set in data_sets:
    #     if data_set != 'sonar':
    #         problem = Problem(data_set, 'pgd', 1, 10,
    #                           lamb_range, 1, 1, 0.01, 1)
    #         problem.benchmark_svm()
    #         problem.benchmark_knn()

    # for degree in range(1, 6):
    #     data_set = 'ionosphere'
    #     problem = Problem(data_set, 'pgd', degree,
    #                       10, lamb_range, 1, 1, 0.01, 1)
    #     problem.cross_validation_sk()
    #     problem.train_test_sk()
    #     problem.plotting_error()
