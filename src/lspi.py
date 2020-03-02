# -*- coding: utf-8 -*-
# agent
from basis_func import *
from policy import *
from collections import namedtuple
import numpy as np
import scipy
from scipy import linalg
import time
from sklearn import linear_model

Transition = namedtuple('Transition',
						('state', 'action','reward', 'next_state', 'done'))


class LSPIAgent(object):
	"""docstring for LSPIAgent"""
	def __init__(self, params):
		super(LSPIAgent, self).__init__()
		self.state_dim = params['state_dim']
		# self.n_actions = params['n_actions']
		self.basisfunc_dim = params['basis_function_dim']
		self.gamma = params['weight_discount']
		self.stop_criterion = params['stop_criterion']
		self.basis_function = params['basis_func'] 
		self.n_basis_func = self.basis_function.size()
		epsilon = 1-params['exploration']
		self.policy = params['policy']
		self.lstdq = LSTDQ(self.basis_function, self.gamma)
		self.n_iter_max = 30

	def train(self, sample, opt='l1'):
		error = float('inf')
		error_his = []
		i_iter = 0
		while error > self.stop_criterion and i_iter<self.n_iter_max:
			new_weights = self.lstdq.training(sample, self.policy, opt)
			# print(new_weights)
			error = np.linalg.norm((new_weights - self.policy.weights))
			# print("error when update_weights in iteration {}: {}".format(i_iter,error))
			error_his.append(error)
			self.policy.update_weights(new_weights)
			i_iter += 1
		# time.sleep(2)
		# print(error_his)
		# print(self.policy.weights)
		
		return error_his, self.policy.weights


	def get_action(self, state):
		return self.policy.get_action_iteracting(state)
		# return self.policy.get_action_LQR(state)


class LSTDQ(object):
	"""docstring for LSTDQ"""
	def __init__(self, basis_function, gamma):
		super(LSTDQ, self).__init__()
		self.basis_function = basis_function
		self.gamma = gamma

	def training(self, samples, policy, opt):
		# Figure 5 in paper
		n = self.basis_function.size()
		A = np.zeros([n, n])
		b = np.zeros([n, 1])
		# l2 weights here:
		np.fill_diagonal(A, .01)  # Singular matrix error

		for s in samples:
			state = s.state
			action = s.action
			reward = s.reward
			next_state = s.next_state
			done = s.done
			# print("state: {}".format(state))
			# print("action: {}".format(action))
			# print("reward: {}".format(reward))
			# print("next_state: {}".format(next_state))
			# print("done: {}".format(done))

			phi = self.basis_function.evaluate(state, action)
			if not done:
				_action = policy.get_action_training(state)
				# print("_action: {}".format(_action))
				phi_next = self.basis_function.evaluate(next_state, _action)
				# print("phi_next: {}".format(phi_next))
			else:
				# print("done")
				phi_next = np.zeros(n)
			loss = phi - self.gamma*phi_next
			phi = np.reshape(phi, [n, 1])
			loss = np.reshape(loss, [1, n])
			# print("phi: {}".format(phi))
			# print("loss: {}".format(loss))
			# print("A: {}".format(A))
			A += np.dot(phi, loss)
			b += phi * reward

		if opt=='l2':
			# \beta of regularization l2 depends on initA
			
			# print("A: {}".format(A))
			# print("b: {}".format(b))
			inv_A = np.linalg.inv(A)
			w = np.dot(inv_A, b)
			print("w: {}".format(w))
			return w
		elif opt=='l1':
			# useing sklearn to solve this
			# importance of regularization l1 is 
			# the alpha in sklearn.linear_model.lasso()
			# if alpha = 0.0, then would be linear regression
			clf = linear_model.Lasso(alpha=1.0,max_iter=10000,tol=0.0001)
			clf.fit(A, b)
			w = np.matrix(clf.coef_).reshape(len(clf.coef_),1)
			print("w: {}".format(w))
			return w




















