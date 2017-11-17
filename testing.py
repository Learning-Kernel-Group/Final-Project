import numpy as np
import pgd as descent


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


if __name__ == '__main__':
    features = np.load(
        './Data Sets/regression-datasets-kin8nm_features_train.npy')
    labels = np.load('./Data Sets/regression-datasets-kin8nm_labels_train.npy')
    testing_features = np.load(
        './Data Sets/regression-datasets-kin8nm_features_test.npy')
    testing_labels = np.load(
        './Data Sets/regression-datasets-kin8nm_labels_test.npy')
    training_result, quad_ker = descent.pgd(
        features, labels, 10, 1, 1, 0.01, np.zeros(8), 1)
    base_kernel_results = hypothesis(features, testing_features, 1)
    predicts = predict(features, labels, training_result,
                       quad_ker, 10, base_kernel_results, testing_features, 1)
    error = error_rate(predicts, testing_labels)
    print(error)
