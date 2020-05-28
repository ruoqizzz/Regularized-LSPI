# -*- coding: utf-8 -*-
import numpy as np
from sklearn.gaussian_process.kernels import RBF
# RBF: Gaussian Multidimensional Radial Basis Function for discrete actions
class RBF(object):
	"""docstring for RBF"""
	def __init__(self, input_dim, n_features, n_actions, sigma, high=np.array([])):
		super(RBF, self).__init__()
		self.input_dim = input_dim
		# print(input_dim)
		self.n_features = n_features
		self.n_actions = n_actions
		self.sigma = sigma
		# print(high)
		if high.size == 0:
			high = np.array([100]*input_dim)
		else:
			assert len(high) == input_dim
		# self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.n_features-1)]
		self.feature_means = np.array([[np.random.uniform(-1*high, high) for _ in range(self.n_features-1)] for _ in range(self.n_actions)])
		# print(self.feature_means[0])


	def size(self):
		return self.n_actions*self.n_features

	def evaluate(self, states, actions):
		k = self.size()
		phi = np.zeros( (states.shape[0], k))
		for i in range(states.shape[0]):
			offset = (self.n_features)*actions[i]
			phi[i, offset] = 1
			rbfs = np.exp(-self.sigma*np.linalg.norm(states[i] - self.feature_means[actions[i]], axis=1)**2)
			print("rbfs.shape: ", rbfs.shape)
			phi[i, offset+1:offset+self.n_features] = rbfs
		return phi

	def name(self):
		return "RBF"


class RBF_LQR(object):
	"""docstring for RBF"""
	def __init__(self, input_dim, n_features, sigma):
		super(RBF_LQR, self).__init__()
		if len(input_dim)==2:
			self.state_dim = input_dim[0]
			self.action_dim = input_dim[1]
		else:
			raise ValueError("len(input_dim) should be 2: state_dim and action_dim")
		self.n_features = n_features
		# TODO sigma -> width
		# keep not so big
		# RANDOM give a range
		self.sigma = sigma
		# the range of mean 
		# self.feature_means = [np.random.uniform([-10,-5],[10,5], (3,2)).transpose((1,0)) for _ in range(self.n_features-1)]
		self.state_means = [ np.random.uniform(-8, 8, self.state_dim) for _ in range(n_features-1)]
		self.action_means = [ np.random.uniform(-4, 4, self.action_dim) for _ in range(n_features-1)]
		# TODO: -2,2 10 
		# TODO: -10,10 10
		# self.feature_means = [np.arange(-2,2,4/(self.n_features-1))]*input_dim
		# self.feature_means = np.arange(n_features)[1:] * 0.1

	def size(self):
		return self.n_features

	def evaluate(self, states, actions):
		k = self.size()
		phi = np.zeros( (states.shape[0], k))
		for i in range(states.shape[0]):
			phi[i,0] = 1
			# print(states[i].tolist())
			# print("self.state_means", self.state_means)
			rbf = np.exp(-self.sigma*(np.linalg.norm(states[i].reshape(1,self.state_dim)[0]- self.state_means, axis=1)**2+np.linalg.norm(actions[i] - self.action_means, axis=1)**2))
			phi[i,1:] = rbf
			# print("rbf.shape: ". rbf.shape)
		return phi

	def name(self):
		return "RBF_LQR"

class Laplace_LQR(object):
	"""docstring for Laplace_LQR"""
	def __init__(self, M, L_vec):
		super(Laplace_LQR, self).__init__()
		self.M = M
		self.L_vec = L_vec
		self.D = len(L_vec)
		self.n_features = self.M**self.D+1

	def size(self):
		return self.n_features

	def evaluate(self, states, actions):
		s = states.reshape(len(actions),len(states[0])).T
		a = np.array(actions).T.reshape(1,len(actions))
		x = np.vstack((s,a))
		phi = np.ones((self.n_features-1, x.shape[1]))
		j_vec = np.ones(self.D)

		for c in range(self.n_features-1):

			for k in range(self.D):
				phi[c, :]  *= np.sin( np.pi*j_vec[k]*( x[k] + self.L_vec[k])/(2*self.L_vec[k]))/np.sqrt(self.L_vec[k])

			j_vec[0] = j_vec[0]+1
			for k in range(1, self.D):
				if (np.mod(j_vec[k-1], self.M+1) == 0):
					j_vec[k-1] = 1
					j_vec[k] = j_vec[k]+1

		return np.vstack(( np.ones(s.shape[1]), phi)).T

	def name(self):
		return "Laplace_LQR"


class ExactBasis4LQR(object):
	"""docstring for ExactBasis4LQR"""
	def __init__(self):
		super(ExactBasis4LQR, self).__init__()
		self.n_features = 4
	
	def size(self):
		return self.n_features

	def evaluate(self, states, actions):
		n = self.size()
		phi = np.zeros((states.shape[0], n))

		for i in range(states.shape[0]):
			s = states[i]
			u = actions[i]
			phi[i,:] = np.array([1, (s.T*s).item(), (u.T*u).item(), (s.T*u).item()])
		# print("in basis evaluate phi_next: {}".format(offset_phi))
		return phi

	def name(self):
		return "ExactBasis_LQR"


# class Polinomial4DiscreteState(object):
# 	"""docstring for Polinomial4DiscreteState"""
# 	def __init__(self, degree,n_actions):
# 		super(Polinomial4DiscreteState, self).__init__()
# 		self.n_actions = n_actions
# 		self.n_features = degree+1	# e.g. 2 [1,s,s**2]
# 		# self.degree = degree
# 	def size(self):
# 		return self.n_features * self.n_actions

# 	def evaluate(self, state, action):
# 		n = self.size()
# 		phi = np.zeros((n, ))
# 		offset = self.n_features*action
		
# 		value = state
# 		offset_phi = [ np.power(value, d) for d in range(self.n_features)]
# 		phi[offset:offset + self.n_features] = offset_phi
# 		return phi
# 		