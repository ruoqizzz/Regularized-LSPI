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
		self.weights = np.random.randn(self.n_basis_func)
		# self.weights = np.array([[0.1]]*self.n_basis_func)
		self.get_action_iteracting = self.get_best_action
		self.get_action_training = self.get_best_action

	def q_state_action_func(self, states, actions):
		# Q(s, a; w) = sum (pi(s, a) * weights)
		# # basis functions pi(s, a)
		phi = self.basis_func.evaluate(states,actions)
		return phi@self.weights  # pi(s, a) * weights

	# epsilon greedy
	def get_best_action_epsilon(self, states):
		if random.random() < self.epsilon:
			rng = np.random.default_rng()
			return rng.choice(self.n_actions, len(states))
		else:
			return self.get_best_action(states)


	def get_best_action(self, states):
		Q_values = np.zeros((states.shape[0], self.n_actions))

		for a in range(self.n_actions):
			Qa = self.q_state_action_func(states, np.full(states.shape[0], a, np.int))
			# print(Qa)
			Q_values[:, a] = Qa
		# print("Q_values: {}".format(Q_values))
		# print(np.argmax(Q_values, axis=1))
		return np.argmax(Q_values, axis=1)

	def update_weights(self, new_weights):
		self.weights = new_weights



class RBFPolicy4LQR(object):
	"""docstring for ExactPolicy4LQR"""
	def __init__(self, basis_func, L = None):
		super(RBFPolicy4LQR, self).__init__()
		if L==None:
			self.get_action_training = self.get_best_action
		else:
			self.L = L
			print("get_action_with_L")
			self.get_action_training = self.get_action_with_L

		self.basis_func = basis_func
		n = basis_func.size()
		self.weights = np.random.randn(n)
		# use the best L 
		self.get_action_iteracting = self.get_best_action


	def q_state_action_func(self, states, actions):
		# Q(s, a; w) = sum (pi(s, a) * weights)
		# # basis functions pi(s, a)
		phi = self.basis_func.evaluate(states,actions)
		# print("phi shape{}".format(phi.shape))
		# print("weights {}".format(self.weights))
		return phi@self.weights  # pi(s, a) * weights


	def get_best_action(self, states):
		actions = np.linspace(-6,6,100)
		Q_values = np.zeros((states.shape[0], len(actions)))

		for i in range(len(actions)):
			Qa = self.q_state_action_func(states, np.full(states.shape[0], actions[i], np.float))
			Q_values[:, i] = Qa
		print("argmax(Q_values: {}".format(np.argmax(Q_values, axis=1)))
		return np.array([actions[i] for i in np.argmax(Q_values, axis=1)])


	def get_action_with_L(self, states):
		print("get_action_with_L")
		actions = []
		for state in states:
			action = - self.L * state
			# print("action: {}".format(action))
			actions.append(action)
		return np.array(actions)

	# action with gaussian noise mean0 noise 1
	def get_best_action_noise(self, state):
		u = self.get_best_action(state)
		m = u.shape[0]
		# noise is scala or vector? 
		return u + np.random.normal(0,1,m)

	def update_weights(self, new_weights):
		self.weights = new_weights


class ExactPolicy4LQR(object):
	"""docstring for ExactPolicy4LQR"""
	def __init__(self, basis_func, L=None):
		super(ExactPolicy4LQR, self).__init__()
		if L==None:
			self.get_action_training = self.get_best_action
		else:
			self.L = L
			self.get_action_training = self.get_action_with_L
		self.basis_func = basis_func
		n = basis_func.size()
		self.weights = np.random.randn(n)
		# use the best L 
		self.get_action_iteracting = self.get_best_action

	def q_state_action_func(self, states, actions):
		# Q(s, a; w) = sum (pi(s, a) * weights)
		# # basis functions pi(s, a)
		phi = self.basis_func.evaluate(states,actions)
		# print("phi shape{}".format(phi.shape))
		# print("weights {}".format(self.weights))
		return phi@self.weights  # pi(s, a) * weights

	def estimate_policy_L(self):
		w3 = self.weights.getA()[2][0]
		w4 = self.weights.getA()[3][0]
		return np.matrix(w4/(2*w3))

	def get_best_action(self, states):
		actions = []
		for s in states:
			# [1, s.T*s, s.T*u, u.T*u] 
			w3 = self.weights[2]
			w4 = self.weights[3]
			action = -w4/(2*w3) * s
			# print("action: {}".format(action))
			actions.append(action)
			# print("L :{}".format(w4/(2*w3)))
			# action = - 0.1 * state
			# print("action: {}".format(action))
		return np.array(actions)

	def get_action_with_L(self, states):
		actions = []
		for s in states:	
			action = - self.L * s
			# print("action: {}".format(action))
			actions.append(action)
		return np.array(actions)

	def update_weights(self, new_weights):
		self.weights = new_weights

	
		

		
