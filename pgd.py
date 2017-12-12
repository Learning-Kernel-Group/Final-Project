# Luca

import numpy as np
from kernel import make_base_kernels

# alg_l2 = Algorithm 2
# Input: 	- xTrain: vector of sample features
#			- yTrain: vector of sample labels
#			- lam: KRR parameter
#			- eta: interpolation parameter
#			- L: optimization problem parameter (Lambda)
#			- mu0: optimization problem parameter
#			- eps: tolerance stopping parameter
#			- mu_init: inital value of mu in the iteration
#			- sumbsampling: sumbsampling factor
# Output: 	- mu_prime: found value of mu
#			- poly_ker: final kernel


def get_base_kernels(features, subsampling=1):
    return make_base_kernels(features, subsampling=subsampling)


def find_kernel(x, y, degree=1, lam=10., eta=0.2, L=1., mu0=None, mu_init=None, eps=1e-3, subsampling=1):
    (m, p) = x.shape
    m = m / subsampling + int(subsampling > 1)
    mu = np.zeros(p)
    mu_prime = mu_init
    base_kernels = get_base_kernels(x, subsampling=subsampling)
    gram = sum_weight_kernels(base_kernels, mu) ** degree + \
        lam * np.eye(m)  # gram = K_mu + lam * I
    y = y[::subsampling]
    al = np.linalg.solve(gram, y)
    it = 0
    it_max = 100
    while np.linalg.norm(mu - mu_prime) > eps and it < it_max:
        mu = mu_prime
        gram = sum_weight_kernels(base_kernels, mu) ** degree + lam * np.eye(m)
        al = np.linalg.solve(gram, y)
        mu_prime = mu + eta * derivatives(degree, base_kernels, mu, al)
        mu = mu0 + L * (mu - mu0) / (np.linalg.norm(mu - mu0) + 1e-9)
        it += 1
    mu = mu_prime
    print 'L = ', L, 'lam = ', lam
    print 'iter = ', it
    base_kernels = get_base_kernels(x, subsampling=1)
    return mu, sum_weight_kernels(base_kernels, mu) ** degree


def derivatives(degree, base_kernels, mu, al):
    d = []
    tmp = sum_weight_kernels(base_kernels, mu)
    for k in range(mu.size):
        center = (tmp ** (degree - 1)) * base_kernels[k, :, :]
        d.append(degree * ((al.T).dot(center)).dot(al))
    return np.array(d)


def sum_weight_kernels(base_kernels, mu):
    tmp = base_kernels.copy()
    for k in range(mu.size):
        tmp[k, :, :] = mu[k] * tmp[k, :, :]
    return np.sum(tmp, 0)
