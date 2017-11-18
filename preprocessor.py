import numpy as np
import csv
from sklearn import preprocessing


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


if __name__ == '__main__':
    preprocessor_data('ionosphere')
    preprocessor_data('sonar')
    preprocessor_csv('regression-datasets-kin8nm')
