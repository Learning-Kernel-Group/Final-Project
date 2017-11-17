import numpy as np


def _build_base_kernel(features, k):
    vec = features[:, k].reshape((-1, 1))
    kernel = vec.dot(vec.T)
    return kernel


def base_kernel_arr(features, subsampling):
    n, p = features.shape
    arr = []
    for k in range(p):
        kernel = _build_base_kernel(features[::subsampling, :], k)
        arr.append(kernel)
    arr = np.array(arr)
    return arr


if __name__ == '__main__':
    training_features = np.load('./Data Sets/regression-datasets-kin8nm_features_train.npy')
    arr = base_kernel_arr(training_features, 100)
