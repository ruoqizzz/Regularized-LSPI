# -*- coding: utf-8 -*-
import numpy as np

# RBF: Gaussian Multidimensional Radial Basis Function
class RBF(object):
	"""docstring for RBF"""
	def __init__(self, input_dim, n_features, n_actions, gamma):
		super(RBF, self).__init__()
		self.input_dim = input_dim
		self.n_features = n_features
		self.n_actions = n_actions
		self.gamma = gamma
		# self.feature_means = [np.random.uniform(-1, 1, input_dim) for _ in range(self.n_features-1)]
		self.feature_means = [[np.random.uniform(-10, 10, input_dim) for _ in range(self.n_features-1)] for _ in range(self.n_actions)]

	def size(self):
		return self.n_actions*self.n_features

	def __calcu_basis_component(self, state, mean, gamma):
		mean_diff = state - mean
		return np.exp(-gamma*np.sum(mean_diff*mean_diff))

	def evaluate(self, state, action):
		k = self.size()
		phi = np.zeros((k,))
		offset = self.n_features * action
		rbf = [self.__calcu_basis_component(state, mean, self.gamma) for mean in self.feature_means[action]]
		phi[offset] = 1.
		phi[offset+1:offset+1+len(rbf)] = rbf
		return phi

	def name(self):
		return "RBF"


class RBF_LQR(object):
	"""docstring for RBF"""
	def __init__(self, input_dim, n_features, gamma):
		super(RBF_LQR, self).__init__()
		if len(input_dim)==2:
			self.state_dim = input_dim[0]
			self.action_dim = input_dim[1]
		else:
			raise ValueError("len(input_dim) should be 2: state_dim and action_dim")
		self.n_features = n_features
		# TODO gamma -> width
		# keep not so big
		# RANDOM give a range
		self.gamma = gamma
		# the range of mean 
		# self.feature_means = [np.random.uniform([-10,-5],[10,5], (3,2)).transpose((1,0)) for _ in range(self.n_features-1)]
		self.state_means = [ np.random.uniform(-10, 10, self.state_dim) for _ in range(n_features-1)]
		self.actions_means = [ np.random.uniform(-6, 6, self.action_dim) for _ in range(n_features-1)]
		# TODO: -2,2 10 
		# TODO: -10,10 10
		# self.feature_means = [np.arange(-2,2,4/(self.n_features-1))]*input_dim
		# self.feature_means = np.arange(n_features)[1:] * 0.1

	def size(self):
		return self.n_features

	def __calcu_basis_component(self,state, action, state_mean, action_mean, gamma):
		state = np.array(state).reshape(1,self.state_dim)[0]
		action = np.array(action).reshape(1,self.action_dim)[0]
		state_diff = state - state_mean
		action_diff = action - action_mean
		return np.exp(-gamma*(np.sum(state_diff*state_diff)+np.sum(action_diff*action_diff)))

	def evaluate(self, state, action):
		n = self.size()
		offset_phi = [self.__calcu_basis_component(state, action, self.state_means[i], self.actions_means[i], self.gamma) for i in range(n-1)]
		return np.array([1] + offset_phi)

	def name(self):
		return "RBF_LQR"

class ExactBasis4LQR(object):
	"""docstring for ExactBasis4LQR"""
	def __init__(self):
		super(ExactBasis4LQR, self).__init__()
		self.n_features = 4
	
	def size(self):
		return self.n_features

	def evaluate(self, state, action):
		n = self.size()
		phi = np.zeros((n, ))
		s = state
		u = action
		# print("BasisFunc evaluate: \n")
		# print("s: {}".format(s))
		# print("u: {}".format(u))

		offset_phi = np.array([1, (s.T*s).item(), (u.T*u).item(), (s.T*u).item()])
		# print("in basis evaluate phi_next: {}".format(offset_phi))
		return offset_phi
	
	def name(self):
		return "ExactBasis_LQR"


class Polinomial4DiscreteState(object):
	"""docstring for Polinomial4DiscreteState"""
	def __init__(self, degree,n_actions):
		super(Polinomial4DiscreteState, self).__init__()
		self.n_actions = n_actions
		self.n_features = degree+1	# e.g. 2 [1,s,s**2]
		# self.degree = degree
	def size(self):
		return self.n_features * self.n_actions

	def evaluate(self, state, action):
		n = self.size()
		phi = np.zeros((n, ))
		offset = self.n_features*action
		
		value = state
		offset_phi = [ np.power(value, d) for d in range(self.n_features)]
		phi[offset:offset + self.n_features] = offset_phi
		return phi
		