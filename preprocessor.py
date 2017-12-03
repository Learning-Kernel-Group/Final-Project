import numpy as np
import csv
from sklearn import preprocessing
import re


def preprocessor_data(filename):
    with open('Data Sets/UCI Data Sets/' + filename + '.data', 'r') as inputfile:
        arr = []
        for line in inputfile:
            item_list = line.rstrip().split(',')
            arr.append(item_list)
        arr = np.array(arr)
        features = arr[:, :-1]
        features = features.astype(np.float)
        labels = arr[:, -1]
        for i in range(labels.shape[0]):
            if labels[i] == 'g':
                labels[i] = 1
            else:
                labels[i] = -1
        labels = labels.astype(np.float)
        n, p = features.shape
        training_features = features[:int(n / 2), :]
        testing_features = features[int(n / 2):, :]
        training_labels = labels[:int(n / 2)]
        testing_labels = labels[int(n / 2):]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit_transform(training_features)
        training_features = scaler.transform(training_features)
        testing_features = scaler.transform(testing_features)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_features_train', training_features)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_features_test', testing_features)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_labels_train', training_labels)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_labels_test', testing_labels)


def preprocessor_csv(filename):
    with open('Data Sets/' + filename + '.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        arr = np.array(data)
        arr = arr.astype(np.float)
        features = arr[:, :-1]
        labels = arr[:, -1]
        mean = np.mean(labels)
        for i in range(labels.shape[0]):
            if labels[i] > mean:
                labels[i] = 1
            else:
                labels[i] = -1
        n, p = features.shape
        training_features = features[:int(n / 2), :]
        testing_features = features[int(n / 2):, :]
        training_labels = labels[:int(n / 2)]
        testing_labels = labels[int(n / 2):]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit_transform(training_features)
        training_features = scaler.transform(training_features)
        testing_features = scaler.transform(testing_features)
        np.save('Data Sets/' + filename + '_features_train', training_features)
        np.save('Data Sets/' + filename + '_features_test', testing_features)
        np.save('Data Sets/' + filename + '_labels_train', training_labels)
        np.save('Data Sets/' + filename + '_labels_test', testing_labels)


def preprocessor_libsvm_data(filename, feature_length, format_label_func=lambda _: _):
    with open('Data Sets/UCI Data Sets/' + filename + '.data', 'r') as inputfile:
        features = []
        labels = []
        for line in inputfile:
            container = line.rstrip().split()
            label = float(container[0])
            label = int(format_label_func(label))
            print(label)
            del container[0]
            pattern = re.compile(r"[-+]?\d+:([-+]?\d*\.\d+|[-+]?\d+)")
            feature = []
            for phrase in container:
                # print(phrase)
                target = re.findall(pattern, phrase)
                # print(target)
                feature.append(float(target[0]))
            if len(feature) == feature_length:
                features.append(feature)
                labels.append(label)
            else:
                print('[WARNING] wrong number of features in data.')
        features = np.array(features)
        labels = np.array(labels)
        n, p = features.shape
        training_features = features[:int(n / 2), :]
        testing_features = features[int(n / 2):, :]
        training_labels = labels[:int(n / 2)]
        testing_labels = labels[int(n / 2):]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit_transform(training_features)
        training_features = scaler.transform(training_features)
        testing_features = scaler.transform(testing_features)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_features_train', training_features)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_features_test', testing_features)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_labels_train', training_labels)
        np.save('Data Sets/UCI Data Sets/' + filename +
                '_labels_test', testing_labels)


def preprocessor_csv2(filename):
    with open('Data Sets/' + filename + '.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        arr = np.array(data)
        arr = arr[1:, :]
        arr = arr.astype(np.float)
        features = arr[:, 1:]
        labels = arr[:, 1]
        mean = np.mean(labels)
        for i in range(labels.shape[0]):
            if labels[i] > mean:
                labels[i] = 1
            else:
                labels[i] = -1
        n, p = features.shape
        training_features = features[:int(n / 2), :]
        testing_features = features[int(n / 2):, :]
        training_labels = labels[:int(n / 2)]
        testing_labels = labels[int(n / 2):]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit_transform(training_features)
        training_features = scaler.transform(training_features)
        testing_features = scaler.transform(testing_features)
        np.save('Data Sets/' + filename + '_features_train', training_features)
        np.save('Data Sets/' + filename + '_features_test', testing_features)
        np.save('Data Sets/' + filename + '_labels_train', training_labels)
        np.save('Data Sets/' + filename + '_labels_test', testing_labels)


if __name__ == '__main__':
    # preprocessor_data('ionosphere')
    # preprocessor_data('sonar')
    # preprocessor_csv('regression-datasets-kin8nm')
    # preprocessor_libsvm_data('breast-cancer', 10, lambda x: x - 3)
    # preprocessor_libsvm_data('diabetes', 8)
    # preprocessor_libsvm_data('fourclass', 2)
    # preprocessor_libsvm_data('german', 24)
    # preprocessor_libsvm_data('heart', 13)
    # preprocessor_libsvm_data('madelon', 500)
    # preprocessor_csv2('supernova')
    features = np.load('Data Sets/supernova_features_train.npy')
    labels = np.load('Data Sets/supernova_labels_train.npy')
    print(features)
    print(labels)
    print(features.shape)
    print(labels.shape)
