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
import cvxpy as cp

Transition = namedtuple('Transition',
						('state', 'action','reward', 'next_state', 'done'))


class LSPIAgent(object):
	"""docstring for LSPIAgent"""
	def __init__(self, params, n_iter_max=50):
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
		self.n_iter_max = n_iter_max
		self.opt = params['reg_opt']
		self.reg_param = params['reg_param']

	def train(self, samples):
		# states = samples[0]
		# print(states[:,2:])
		states = samples[0][:,2:]
		actions = samples[1]
		# rewards = samples[2]
		rewards = -(states[:,0]**2)
		# next_states = samples[3]
		next_states = samples[3][:,2:]
		dones = samples[4]

		phi = self.policy.basis_func.evaluate(states, actions)

		error = float('inf')
		error_his = []
		i_iter = 0
		while error > self.stop_criterion and i_iter<self.n_iter_max:
			i_iter += 1
			next_actions = self.policy.get_action_training(next_states)
			# print(self.policy.basis_func.evaluate(next_states, next_actions)[:10])
			# print((self.policy.basis_func.evaluate(next_states, next_actions)*(1-dones).reshape(len(dones),1))[:10])
			# print(np.array(dones).astype(float))
			next_phi = self.policy.basis_func.evaluate(next_states, next_actions)
			# *(1.0-np.array(dones).astype(float)).reshape(len(dones),1)
			A = 1/states.shape[0]*phi.T@(phi-self.gamma*next_phi) 
			# print("A: {}".format(A))
			if self.opt=='l2':
				A += self.reg_param*np.identity(A.shape[0])
			else:
				A += 0.0001*np.identity(A.shape[0])
			# 1/states.shape[0]
			# A = 1/states.shape[0]* A
			b = 1/states.shape[0]*phi.T@rewards
			# print("b: {}".format(b))
			
			if self.opt == 'l1':
				clf = linear_model.Lasso(alpha=self.reg_param, max_iter=50000)
				clf.fit(A, b)
				new_weights = clf.coef_
			if self.opt == 'l2':
				new_weights = np.linalg.solve(A,b)
			error = np.linalg.norm(self.policy.weights-new_weights)
			print("error when update_weights in iteration {}: {}".format(i_iter,error))
			if len(error_his)>2:
				if error == error_his[-1] and error == error_his[-2]:
					print("Weights jump between two values, break")
					self.policy.update_weights(new_weights)
					break;
			error_his.append(error)
			# print("new_weights: {}".format(new_weights))
			self.policy.update_weights(new_weights)
		return error_his, self.policy.weights

	def get_action(self, state):
		return self.policy.get_action_iteracting(state)

