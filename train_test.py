import numpy as np
import pgd as descent
from sklearn.model_selection import RepeatedKFold


def hypothesis(training_features, testing_features, subsampling):
    training_features = training_features[::subsampling, :]
    base_kernel_results = []
    _, p = training_features.shape
    for k in range(p):
        training_vec = training_features[:, k]
        testing_vec = testing_features[:, k]
        kernel_hypothesis = (training_vec.reshape(
            (-1, 1))).dot(testing_vec.reshape((1, -1)))
        base_kernel_results.append(kernel_hypothesis)
    return np.array(base_kernel_results)


# def _apply_base_kernel(vec1, vec2, i):
#     return vec1[i] * vec2[i]


def predict(training_features, training_labels, training_result, quad_ker, lamb, base_kernel_results, testing_features, subsampling):
    base_kernel_results = hypothesis(
        training_features, testing_features, subsampling)
    _, p = training_features.shape
    labels = training_labels[::subsampling]
    weighted_kernels = descent._weighting_kernels(
        base_kernel_results, training_result, p)
    sum_ker = np.sum(weighted_kernels, 0)
    cross_ker = sum_ker ** 2
    inverse = descent._inverse_part(quad_ker, lamb)
    predicts = (labels.dot(inverse)).dot(cross_ker)
    return np.sign(predicts)


def error_rate(predicts, labels):
    error_rate_1 = np.linalg.norm((predicts - labels) / 2, 1)
    error_rate_2 = np.linalg.norm((predicts + labels) / 2, 1)
    return min(error_rate_1, error_rate_2) / labels.shape[0]


def mean_squared_error(predicts, labels):
    n = labels.shape
    mean_squared_sum = np.sum((predicts - labels) ** 2) / 4
    return mean_squared_sum / n


def cross_validation(features, labels, folds, method, lamb, eta, norm_bound, tolerence, mu_0, subsampling):
    rkf = RepeatedKFold(n_splits=folds, n_repeats=1)
    error_arr = []
    mse_arr = []
    for train_index, test_index in rkf.split(features):
        features_train, features_test = features[
            train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        training_result, quad_ker = training(
            features_train, labels_train, lamb, eta, norm_bound, tolerence, mu_0, subsampling, method)
        base_kernel_results = hypothesis(
            features_train, features_test, subsampling)
        predicts = predict(features_train, labels_train, training_result,
                           quad_ker, lamb, base_kernel_results, features_test, subsampling)
        error = error_rate(predicts, labels_test)
        mse = mean_squared_error(predicts, labels_test)
        error_arr.append(error)
        mse_arr.append(mse)
    return np.array(error_arr), np.array(mse_arr)


def training(features, labels, lamb, eta, norm_bound, tolerence, mu_0, subsampling, method='pgd'):
    if method == 'pgd':
        training_result, quad_ker = descent.pgd(
            features, labels, lamb, eta, norm_bound, tolerence, mu_0, subsampling)
        return training_result, quad_ker


def testing(training_result, quad_ker, features, labels, testing_features, testing_labels, lamb, eta, norm_bound, tolerence, mu_0, subsampling):
    n, p = features.shape
    base_kernel_results = hypothesis(features, testing_features, subsampling)
    predicts = predict(features, labels, training_result,
                       quad_ker, lamb, base_kernel_results, testing_features, subsampling)
    error = error_rate(predicts, testing_labels)
    mse = mean_squared_error(predicts, testing_labels)
    return error, mse


if __name__ == '__main__':
    # features = np.load(
    #     './Data Sets/regression-datasets-kin8nm_features_train.npy')
    # labels = np.load('./Data Sets/regression-datasets-kin8nm_labels_train.npy')
    # testing_features = np.load(
    #     './Data Sets/regression-datasets-kin8nm_features_test.npy')
    # testing_labels = np.load(
    #     './Data Sets/regression-datasets-kin8nm_labels_test.npy')
    # n, p = features.shape
    # training_result, quad_ker = training(features, labels, 10, 1, 1, 0.01, np.zeros(p), 10, method='pgd')
    # error = testing(training_result, quad_ker, features, labels, testing_features,
    #                 testing_labels, 10, 1, 1, 0.01, np.zeros(p), 10)
    # error_arr = cross_validation(features, labels, 10, 10, 1, 1, 0.01, np.zeros(p), 10)
    # print(error)
    # print(error_arr)

    features = np.load(
        './Data Sets/UCI Data Sets/ionosphere_features_train.npy')
    labels = np.load('./Data Sets/UCI Data Sets/ionosphere_labels_train.npy')
    testing_features = np.load(
        './Data Sets/UCI Data Sets/ionosphere_features_test.npy')
    testing_labels = np.load(
        './Data Sets/UCI Data Sets/ionosphere_labels_test.npy')
    n, p = features.shape
    training_result, quad_ker = training(
        features, labels, 10, 1, 1, 0.01, np.zeros(p), 1, method='pgd')
    error = testing(training_result, quad_ker, features, labels, testing_features,
                    testing_labels, 10, 1, 1, 0.01, np.zeros(p), 1)
    error_arr = cross_validation(
        features, labels, 10, 10, 1, 1, 0.01, np.zeros(p), 1)
    print(error)
    print(error_arr)

    # features = np.load(
    #     './Data Sets/UCI Data Sets/sonar_features_train.npy')
    # labels = np.load('./Data Sets/UCI Data Sets/sonar_labels_train.npy')
    # testing_features = np.load(
    #     './Data Sets/UCI Data Sets/sonar_features_test.npy')
    # testing_labels = np.load(
    #     './Data Sets/UCI Data Sets/sonar_labels_test.npy')
    # n, p = features.shape
    # training_result, quad_ker = training(features, labels, 10, 1, 1, 0.01, np.zeros(p), 1, method='pgd')
    # error = testing(training_result, quad_ker, features, labels, testing_features,
    #                 testing_labels, 10, 1, 1, 0.01, np.zeros(p), 1)
    # error_arr = cross_validation(features, labels, 10, 10, 1, 1, 0.01, np.zeros(p), 10)
    # print(error)
    print(error_arr)
