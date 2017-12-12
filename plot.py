from matplotlib import pyplot as plt
import numpy as np


def plot_as_seq(arr, xaxis, label, ax):
    ax.plot(xaxis, arr, marker='D', label=label)


def plot_as_errorbar(trails, xaxis, label, ax):
    mean = np.mean(trails, axis=1)
    deviation = np.std(trails, axis=1)
    ax.errorbar(xaxis, mean, yerr=deviation, fmt='-.D', label=label)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arr = [1, 2, 3, 4, 5, 4, 3, 27, 8, 8, 9, 6]
    arr = np.array(arr)
    xaxis = np.array(range(1, 13))
    plot_as_seq(arr, xaxis, ax)
    arr = [[1,20],[3,30]]
    xaxis = [1,2]
    plot_as_errorbar(arr, xaxis, ax)
    # plt.savefig('imagefile.png', dpi=200)
    plt.show()
