import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pgd_l2
import pgd_new
import pgd
import kernel


class Problem():

    def __init__(self, dataset=None, alg=None, method=None, degree=1, c_range=[1.],
                 lam_range=[1.], eta=1., L_range=[1.], mu0=1., mu_init=1., eps=1e-3, subsampling=1):

        self.dataset = dataset
        self.find_kernel = eval(alg).find_kernel
        self.sum_weight_kernels = eval(alg).sum_weight_kernels
        self.method = method
        self.degree = degree
        self.c_range = c_range
        self.lam_range = lam_range
        self.eta = eta
        self.L_range = L_range
        with open('data_python/' + self.dataset, 'rb') as f:
            [self.xTrain, self.yTrain, self.xTest,
                self.yTest] = pickle.load(f)
        self.n_features = self.xTrain.shape[1]
        self.mu0 = mu0 * np.ones(self.n_features)
        self.mu_init = mu_init * np.ones(self.n_features)
        self.eps = eps
        self.subsampling = subsampling
        self.best_L = None
        self.best_lam = None
        self.best_c = None
        self.model = None
        self.mu = None
        self.make_test_kernels = kernel.make_test_kernels
        self.cv_error = np.empty((len(L_range), len(lam_range), len(c_range)))
        self.cv_error_best = None
        self.test_error = None

    def get_classifier(self, c=1.):
        if self.method == 'SVC':
            return SVC(C=c, kernel='precomputed')
        if self.method == 'KRR':
            return KernelRidge(alpha=c, kernel='precomputed')

    def get_kernel(self, lam=1., L=1.):
        return self.find_kernel(self.xTrain, self.yTrain, degree=self.degree, lam=lam, eta=self.eta,
                                L=L, mu0=self.mu0, mu_init=self.mu_init, eps=self.eps, subsampling=self.subsampling)

    def cv(self):
        for L in range(len(self.L_range)):
            for lam in range(len(self.lam_range)):
                _, gTrain = self.get_kernel(lam=lam_range[lam], L=L_range[L])
                for c in range(len(self.c_range)):
                    classifier = self.get_classifier(c=c_range[c])
                    self.cv_error[
                        L, lam, c] = 1. - cross_val_score(classifier, gTrain, self.yTrain, cv=10).mean()
                    print('c = ', c_range[c], ' -> ', self.cv_error[L, lam, c])
        self.best_L, self.best_lam, self.best_c = np.unravel_index(self.cv_error.argmin(), self.cv_error.shape)
        self.cv_error_best = self.cv_error[
            self.best_L, self.best_lam, self.best_c]
        classifier = self.get_classifier(c=self.c_range[self.best_c])
        self.mu, self.gTrain = self.get_kernel(
            lam=self.lam_range[self.best_lam], L=self.L_range[self.best_L])
        self.model = classifier.fit(gTrain, self.yTrain)

    def score(self):
        tmp = self.make_test_kernels(
            self.xTrain, self.xTest, subsampling=self.subsampling)
        self.gTest = self.sum_weight_kernels(tmp, self.mu) ** self.degree
        self.test_error = 1. - self.model.score(self.gTest, self.yTest)

    def mse(self):
        self.mse_cv_error = np.sqrt(mean_squared_error(
            self.yTrain, self.model.predict(self.gTrain)))
        self.mse_test_error = np.sqrt(mean_squared_error(
            self.yTest, self.model.predict(self.gTest)))

    def benchmark(self, method=None):
        print('benchmark model: ' + method)
        classifier = eval(method)
        print('cv error -> ', 1. - cross_val_score(classifier, self.xTrain, self.yTrain, cv=5).mean())
        classifier.fit(self.xTrain, self.yTrain)
        print ('test error -> ', 1. - classifier.score(self.xTest, self.yTest))

if __name__ == '__main__':

    data_sets = {1: 'ionosphere', 2: 'sonar', 3: 'breast-cancer', 4: 'diabetes', 5: 'fourclass', 6: 'german',
                 7: 'heart', 8: 'kin8nm', 9: 'madelon', 10: 'supernova'}

    data = 1
    alg = 'pgd'
    method = 'KRR'
    degree = 2
    c_range = [2 ** i for i in [-8, -4, -2, 0, 2, 4, 8]]
    lam_range = [0.01, 0.1, 1., 10., 50., 100.]
    eta = 0.6
    L_range = [1., 10., 50., 100.]
    eps = 1e-3
    subsampling = 100
    mu0 = 1.
    mu_init = 1.

    dataset = data_sets[data]
    problem = Problem(dataset=dataset, alg=alg, method=method, degree=degree,
                      c_range=c_range, lam_range=lam_range, eta=eta, L_range=L_range, mu0=mu0, mu_init=mu_init, eps=eps, subsampling=subsampling)
    problem.cv()
    print ('cv -> ', problem.cv_error_best)
    problem.score()
    print ('test -> ', problem.test_error)
    problem.mse()
    print ('mse cv -> ', problem.mse_cv_error)
    print ('mse test -> ', problem.mse_test_error)
    problem.benchmark(method='KernelRidge()')
