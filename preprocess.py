import numpy as np
import pickle
from sklearn import preprocessing
import re


def _preprocess(data_set):
    with open('data/uci/' + data_set + '.data', 'r') as _file:
        data_array = []
        for line in _file:
            tmp = line.rstrip().split(',')
            data_array.append(tmp)
        data_array = np.array(data_array)
        np.random.shuffle(data_array)
        features = data_array[:, :-1]
        features = features.astype(np.float)
        labels = data_array[:, -1]
        classes = list(set(labels))
        if data_set != 'kin8nm':
            for i in range(labels.shape[0]):
                if labels[i] == classes[0]:
                    labels[i] = 1
                else:
                    labels[i] = -1
        labels = labels.astype(np.float)  # sure ?
        frac = int(features.shape[0] * 0.8)
        xTrain = features[:frac, :]
        xTest = features[frac:, :]
        yTrain = labels[:frac]
        yTest = labels[frac:]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1., 1.))
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        _list = [xTrain, yTrain, xTest, yTest]
        with open('data_python/' + data_set, 'wb') as _file:
            pickle.dump(_list, _file)


def preprocessor_libsvm_data(filename, format_label_func=lambda _: _):
    with open('data/uci/' + filename + '.data', 'r') as inputfile:
        features = []
        labels = []
        for line in inputfile:
            container = line.rstrip().split()
            label = float(container[0])
            label = int(format_label_func(label))
            del container[0]
            pattern = re.compile(r"[-+]?\d+:([-+]?\d*\.\d+|[-+]?\d+)")
            feature = []
            for phrase in container:
                # print(phrase)
                target = re.findall(pattern, phrase)
                # print(target)
                feature.append(float(target[0]))
            features.append(feature)
            labels.append(label)
        features = np.array(features)
        labels = np.array(labels).reshape((-1, 1))
        data = np.concatenate((features, labels), axis=1)
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1]
        frac = int(features.shape[0] * 0.8)
        xTrain = x[:frac, :]
        xTest = x[frac:, :]
        yTrain = y[:frac]
        yTest = y[frac:]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1., 1.))
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        _list = [xTrain, yTrain, xTest, yTest]
        with open('data_python/' + filename, 'wb') as _file:
            pickle.dump(_list, _file)


def _load_and_save(dataset):
    xTrain = np.load('data/data_chris/' + dataset + '_features_train.npy')
    yTrain = np.load('data/data_chris/' + dataset + '_labels_train.npy')
    xTest = np.load('data/data_chris/' + dataset + '_features_test.npy')
    yTest = np.load('data/data_chris/' + dataset + '_labels_test.npy')
    x = np.concatenate((xTrain, xTest), axis=0)
    y = np.concatenate((yTrain, yTest), axis=0).reshape((-1, 1))
    data = np.concatenate((x, y), axis=1)
    _preprocess()

if __name__ == '__main__':
    data_sets = ['ionosphere', 'sonar', 'kin8nm']
    for dataset in data_sets:
        _preprocess(dataset)
    data_sets_chris = ['breast-cancer']  # , 'diabetes', 'fourclass', 'german',
    #'heart', 'kin8nm', 'madelon', 'supernova']
    for dataset in data_sets_chris:
        preprocessor_libsvm_data(dataset)
