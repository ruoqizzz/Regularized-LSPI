# -*- coding: utf-8 -*-
# policy greedy
import numpy as np
import random
from numpy import matlib as mb

class GreedyPolicy(object):
	"""docstring for GreedyPolicy"""
	def __init__(self, basis_func, n_actions, epsilon):
		super(GreedyPolicy, self).__init__()
		self.basis_func = basis_func
		self.epsilon = epsilon
		self.n_basis_func = self.basis_func.size()
		self.n_actions = n_actions
		self.actions = list(range(n_actions))
		self.weights = np.random.uniform(-1.0, 1.0, size=(self.n_basis_func, 1))


	def q_state_action_func(self, state, action):
        # Q(s, a; w) = sum (pi(s, a) * weights)
        # # basis functions pi(s, a)
		vector_basis = self.basis_func.evaluate(state,action)
		return np.dot(vector_basis, self.weights)  # pi(s, a) * weights

	def get_best_action_epsilon(self, state):
		q_state_actions = [self.q_state_action_func(state, a) for a in self.actions]
		q_state_actions = np.reshape(q_state_actions, [len(q_state_actions), 1]) # convert to column vector
		index = np.argmax(q_state_actions)
		q_max = q_state_actions[index]
		best_action = self.actions[index]
		rng = np.random.default_rng()
		# epsilon greedy
		if rng.random() < self.epsilon:
			return self.actions[index]
		else:
			# print(self.actions)
			return random.sample(self.actions,1)[0]

	def get_best_action(self, state):
		q_state_actions = [self.q_state_action_func(state, a) for a in self.actions]
		q_state_actions = np.reshape(q_state_actions, [len(q_state_actions), 1]) # convert to column vector
		index = np.argmax(q_state_actions)
		q_max = q_state_actions[index]
		best_action = self.actions[index]
		rng = np.random.default_rng()
		return self.actions[index]


	def update_weights(self, new_weights):
		self.weights = new_weights


class ExactPolicy4LQR(object):
	"""docstring for ExactPolicy4LQR"""
	def __init__(self, basis_func, L=np.matrix(0.3), epsilon=0.1):
		super(ExactPolicy4LQR, self).__init__()
		self.L = L
		self.epsilon = epsilon
		self.basis_func = basis_func
		n = basis_func.size()
		self.weights = mb.rand(n,1)

	def q_state_action_func(self, state, action):
		basis = self.basis_func.evaluate(state,action)
		return basis.T * self.weights


	def get_best_action(self, state):
		# [1, s.T*s, s.T*u, u.T*u] 
		print("state: {}".format(state))
		print("weights: {}".format(self.weights))
		w3 = self.weights.getA()[2][0]
		w4 = self.weights.getA()[3][0]
		action = -w4/(2*w3) * state
		# action = - 0.1 * state
		print("action: {}".format(action))
		return action

	def get_action_with_L(self, state, L):
		# [1, s.T*s, s.T*u, u.T*u] 
		print("state: {}".format(state))
		print("weights: {}".format(self.weights))
		w3 = self.weights.getA()[2][0]
		w4 = self.weights.getA()[3][0]
		action = - L * state
		print("action: {}".format(action))
		return action.item()

	def get_best_action_noise(self, state):
		u = self.get_best_action(state)
		m = u.shape[0]
		# noise is scala or vector? 
		return u + np.random.normal(0,1,m)

	def update_weights(self, new_weights):
		self.weights = np.matrix(new_weights)

	
		

		
