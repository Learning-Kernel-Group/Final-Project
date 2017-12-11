import numpy as np

def build_base_kernel(features):
    v = features.reshape((-1, 1))
    return v.dot(v.T)

def build_test_kernel(xTrain, xTest):
    vTrain = xTrain.reshape((-1, 1))
    vTest = xTest.reshape((-1, 1))
    return vTest.dot(vTrain.T)

def make_base_kernels(features, subsampling=1):
    kernels_array = []
    for k in range(features.shape[1]):
        tmp = build_base_kernel(features[::subsampling, k])
        kernels_array.append(tmp)
    return np.array(kernels_array)

def make_test_kernels(xTrain, xTest, subsampling=1):
    kernels_array = []
    for k in range(xTrain.shape[1]):
        tmp = build_test_kernel(xTrain[:, k], xTest[:, k])
        #tmp = build_test_kernel(xTrain[::subsampling, k], xTest[::subsampling, k])
        kernels_array.append(tmp)
    return np.array(kernels_array)
