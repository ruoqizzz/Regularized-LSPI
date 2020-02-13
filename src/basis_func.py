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
		self.feature_means = np.arange(n_features)[1:] * 0.1

	def size(self):
		return self.n_actions*self.n_features

	def __calcu_basis_component(self,state, mean, gamma):
		mean_diff = state - mean
		return np.exp(-gamma*np.sum(mean_diff*mean_diff))

	def evaluate(self, state, action):
		k = self.size()
		phi = np.zeros((k,))
		offset = self.n_features * action
		rbf = [self.__calcu_basis_component(state, mean, self.gamma) for mean in self.feature_means]
		phi[offset] = 1.
		phi[offset+1:offset+1+len(rbf)] = rbf
		return phi

class ExactFunc4LQR(object):
	"""docstring for ExactFunc4LQR"""
	def __init__(self, n_actions):
		super(ExactFunc4LQR, self).__init__()
		self.n_features = 4
		self.n_actions = n_actions
	

	def size(self):
		return self.n_features * self.n_actions

	def evaluate(self, state, action):
		n = self.size()
		phi = np.zeros((n, ))

        offset = (n/self.num_actions)*action
        
        value = state

        offset_phi = np.array([1, pow(value,2), pow(action,2), value*action])

        phi[offset:offset + self.degree + 1] = offset_phi

        return phi




		