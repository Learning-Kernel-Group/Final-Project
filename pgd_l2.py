# Luca

# inter = Algorithm 2
# Input: 	- features: vector of sample features
#			- labels: vector of sample labels
#			- lam: KRR parameter
#			- eta: interpolation parameter
#			- L: optimization problem parameter (Lambda)
#			- mu0: optimization problem parameter
#			- eps: tolerance stopping parameter
#			- mu_init: inital value of mu in the iteration
#			- sumbsampling: sumbsampling factor
# Output: 	- mu_prime: found value of mu
#			- poly_ker: final kernel

def inter(features, labels, degree=1, lam=10., eta=0.5, L=1., mu0=None, eps=1e-3, mu_init=None, subsampling=1):
	(m, p) = features.shape
    mu = mu_init
	if mu == None:
		mu = np.zeros(p)
	if mu0 == None:
		mu0 = np.zeros(p)    
	print('Start running my_int on kernels...')
	gram = sum_weight_ker(base_kernels, mu) + lam * np.eye(m) # gram = K_mu + lam * I
	print('Solve for alpha...')
	al = np.linalg.solve(gram, labels)
	al_prime = np.zeros(m)
    print('Start interpolated algorithm loop...')
    while np.linalg.norm(al - al_prime) > eps:
        al = al_prime
        print('Computing new weights...')
        v = _derivatives(base_kernels, mu, al) ##
		v /= np.linalg.norm(v)		
		mu = mu0 + L * v 
        print('Computing new alpha...')
		gram = sum_weigh_ker(base_kernels, mu) + lam * np.eye(m)
        al_prime = eta * al + (1. - eta) * np.linalg.solve(gram,labels)
        print('The weight vector for this round is:\n', mu)
    return sum_weight_ker(base_kernels, mu)

def _derivatives(degree, base_kernels, mu, al):
    derivatives = []
	sum_ker = sum_weight_kernels(base_kernels, mu)
    for k in range(p):
        derivatives.append(((al.T).dot(sum_ker)).dot(al))
    return np.array(derivatives)

def sum_weight_kernels(base_kernels, mu):
    for k in range(mu.size):
        base_kernels[k, :, :] = mu[k] * base_kernels[k, :, :]
    return np.sum(base_kernels, 0)
