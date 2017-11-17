import numpy as np
import csv


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
        np.save('Data Sets/UCI Data Sets/' + filename + '_features', features)
        np.save('Data Sets/UCI Data Sets/' + filename + '_labels', labels)


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
        training_features = features[:4096, :]
        testing_features = features[4096: , :]
        training_labels = labels[:4096]
        testing_labels = labels[4096:]
        np.save('Data Sets/' + filename + '_features_train', training_features)
        np.save('Data Sets/' + filename + '_features_test', testing_features)
        np.save('Data Sets/' + filename + '_labels_train', training_labels)
        np.save('Data Sets/' + filename + '_labels_test', testing_labels)


if __name__ == '__main__':
    # preprocessor_data('ionosphere')
    # preprocessor_data('sonar')
    preprocessor_csv('regression-datasets-kin8nm')
