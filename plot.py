from matplotlib import pyplot as plt
import numpy as np


def plot_as_seq(arr, ax):
    n = arr.shape[0]
    xaxis = range(1, n + 1)
    ax.plot(xaxis, arr)


def plot_as_errorbar(trails, xaxis, ax):
    mean = np.mean(trails, axis=1)
    deviation = np.std(trails, axis=1)
    ax.errorbar(xaxis, mean, yerr=deviation)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arr = [1, 2, 3, 4, 5, 4, 3, 27, 8, 8, 9, 6]
    arr = np.array(arr)
    plot_as_seq(arr, ax)
    arr = [[1,2],[3,4]]
    xaxis = [1,2]
    plot_as_errorbar(arr, xaxis, ax)
    plt.savefig('imagefile.png', dpi=200)
    plt.show()
