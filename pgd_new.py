# Luca

# my_int = Algorithm 3
# Input: 	- features: vector of sample features
#			- labels: vector of sample labels
#			- degree: degree of polynomial combination
#			- lamb: KRR parameter
#			- eta: interpolation parameter
#			- beta: optimization problem parameter
#			- eps: tolerance stopping parameter
#			- mu_init: inital value of mu in the iteration
#			- sumbsampling: sumbsampling factor
# Output: 	- mu_prime: found value of mu
#			- poly_ker: final kernel

def my_int(features, labels, degree=1, lam=10., eta=0.5, beta=1., eps=1e-3, mu_init=None, subsampling=1):
	beta = 0.5 / beta
    (m, p) = features.shape
    mu = mu_init
	if mu == None:
		mu = np.zeros(p)    
	print('Start running my_int on kernels...')
	gram = sum_weight_ker(base_kernels, mu) + lam * np.eye(m) # gram = K_mu + lam * I
	print('Solve for alpha...')
	al = np.linalg.solve(gram, labels)
	al_prime = np.zeros(m)
    print('Start interpolated algorithm loop...')
    while np.linalg.norm(al - al_prime) > eps:
        al = al_prime
        print('Computing new weights...')
        mu = beta * _derivatives(degree, base_kernels, mu, al) ##
        print('Computing new alpha...')
		gram = sum_weigh_ker(base_kernels, mu) + lam * np.eye(m)
        al_prime = eta * al + (1. - eta) * np.linalg.solve(gram,labels)
        print('The weight vector for this round is:\n', mu)
    poly_ker = sum_weight_ker(base_kernels, mu) ** degree
    return mu, poly_ker

# why don't we just solve the linear problem? instead of inverse

def _derivatives(degree, base_kernels, mu, al):
    derivatives = []
	sum_ker = sum_weight_kernels(base_kernels, mu)
    for k in range(p):
        center = (sum_ker ** (degree -1)) * base_kernels[k]
        derivatives.append(degree * ((al.T).dot(center)).dot(al))
    return np.array(derivatives)

def sum_weight_kernels(base_kernels, mu):
    for k in range(mu.size):
        base_kernels[k, :, :] = mu[k] * base_kernels[k, :, :]
    return np.sum(base_kernels, 0)
