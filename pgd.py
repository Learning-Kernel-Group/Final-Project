import numpy as np
import kernel


def _get_base_kernels(features, subsampling=1):
    return kernel.base_kernel_arr(features, subsampling)


def pgd(features, labels, degree, lamb, eta, norm_bound, tolerence, mu_0, subsampling):
    print('Start running pgd on kernels...')
    n, p = features.shape
    mu = np.zeros(p)
    mu_prime = np.ones(p)
    print('Start getting base kernels...')
    base_kers = _get_base_kernels(features, subsampling)
    labels = labels[::subsampling]
    quad_ker = np.empty(1)
    print('Start projection-based gradient descent loop...')
    while np.linalg.norm(mu - mu_prime) > tolerence:
        mu = mu_prime
        print('Computing weighted kernels...')
        weighted_kernels = _weighting_kernels(base_kers, mu, p)
        sum_ker = np.sum(weighted_kernels, 0)
        quad_ker = sum_ker ** degree
        print('Computing derivatives of the objective function...')
        derivatives = _partial_derivatives(
            quad_ker, sum_ker, base_kers, lamb, labels, p)
        print('Updating the weight vector...')
        mu_prime = mu - eta * derivatives
        mu_prime = _normalization(mu_prime, mu_0, norm_bound)
        print('The weight vector for this round is:\n', mu_prime)
    weighted_kernels = _weighting_kernels(base_kers, mu_prime, p)
    sum_ker = np.sum(weighted_kernels, 0)
    poly_ker = sum_ker ** degree
    return mu_prime, poly_ker


def _normalization(mu_prime, mu_0, norm_bound):
    difference = mu_prime - mu_0
    difference = norm_bound * difference / np.linalg.norm(difference)
    return difference + mu_0


def _partial_derivatives(quad_ker, sum_ker, base_kernels, lamb, labels, p):
    derivatives = []
    for k in range(p):
        center = (sum_ker ** (degree-1)) * base_kernels[k] # sum_ker * base_kernels[k]
        print('Inverting matrix...')
        inverse = _inverse_part(quad_ker, lamb)
        edge_part = inverse.dot(labels)
        derivatives.append(-degree * ((edge_part.T).dot(center)).dot(edge_part)) # -2 instead of -degree
    return np.array(derivatives)


def _inverse_part(quad_ker, lamb):
    mat = quad_ker + lamb * np.eye(quad_ker.shape[0])
    return np.linalg.inv(mat)


def _weighting_kernels(base_kernels, mu, p):
    for k in range(p):
        base_kernels[k, :, :] = mu[k] * base_kernels[k, :, :]
    return base_kernels


if __name__ == '__main__':
    # a = [1, 2, 3]
    # b = [[[5, 5], [5, 5]], [[7, 7], [7, 7]], [[9, 9], [9, 9]]]
    # a = np.array(a)
    # b = np.array(b)
    # for i in range(a.shape[0]):
    #     b[i,:,:] = a[i] * b[i,:,:]
    # print(b)
    # b = np.sum(b, 0)
    # print(b)
    features = np.load('./Data Sets/regression-datasets-kin8nm_features_train.npy')
    labels = np.load('./Data Sets/regression-datasets-kin8nm_labels_train.npy')
    training_result, quad_ker = pgd(features, labels, 10, 1, 1, 0.01, np.zeros(8), 100)
