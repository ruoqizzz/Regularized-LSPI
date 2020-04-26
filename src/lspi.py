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
		self.old_weights = None

	def train(self, samples):
		# states = samples[0]
		# print(states[:,2:])
		# states = samples[0][:,2:]
		# print(states.shape)
		states = samples[0]
		actions = samples[1]
		# rewards = samples[2]
		rewards = -(states[:,2]**2 + states[:,0]**2)
		# rewards = -(states[:,2]**2)
		# rewards = -(states[:,0]**2)
		# print(rewards.shape)
		next_states = samples[3]
		# next_states = samples[3][:,2:]
		dones = samples[4]
		
		phi = self.policy.basis_func.evaluate(states, actions)
		# print("shape of phi: ", phi.shape)
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
			# print("shape of next_phi: ", next_phi.shape)
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
			# print("shape of b: ", b.shape)
			# print("shape of A: ", A.shape)
			if self.opt == 'l1':
				clf = linear_model.Lasso(alpha=self.reg_param, max_iter=50000)
				clf.fit(A, b)
				new_weights = clf.coef_
			elif self.opt == 'l2':
				new_weights = np.linalg.solve(A,b)
			elif self.opt == 'wl1':
				# TODO: test
				n = A.shape[1]
				lambdas = np.sqrt(np.diag(A.T@A)/A.shape[0])
				lambdas = np.diag(lambdas)
				if self.old_weights is None:
					new_weights = np.zeros(A.shape[1])
				else:
					new_weights = self.old_weights
				# weights_his = []
				T = A.T@A
				kepa = np.linalg.norm(b, ord=2)**2
				rho = A.T@b
				eta = np.linalg.norm(b-A@new_weights, ord=2)**2
				zeta = A.T@(b-A@new_weights)
				k=0
				ifChange = True
				while(ifChange):
					ifChange = False
					for i in range(n):
						# get w/i 
						beta_i = T[i][i]
						alpha_i = eta + beta_i*new_weights[i]**2 + 2*zeta[i]*new_weights[i]
						g_i = zeta[i] + beta_i*new_weights[i]
						l_i = lambdas[i,i]
						r_i_hat = 0
						if alpha_i*(l_i**2)<g_i**2:
							r_i_hat = abs(g_i)/beta_i - l_i/(beta_i*np.sqrt(beta_i-l_i**2))*np.sqrt(alpha_i*beta_i-g_i**2)
				#         if w[i]-np.sign(g_i)*r_i_hat>10e-20:
						old_wi = new_weights[i]
						new_weights[i] = np.sign(g_i)*r_i_hat
						diff = old_wi - new_weights[i]
						if diff>10e-3:
							ifChange = True
						eta += beta_i*diff**2 + 2*diff*zeta[i] 
						zeta += T[:,i]*diff
					k+=1
					# print(k)
					self.old_weights = new_weights
					# weights_his.append(w)
			else:
				assert ValueError("wrong option")
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
		# print("error_his: ",error_his)
		return error_his, self.policy.weights

	def get_action(self, state):
		return self.policy.get_action_iteracting(state)

