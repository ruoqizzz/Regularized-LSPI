# agent
from basis_func import RBF
from policy import GreedyPolicy
from collections import namedtuple
import numpy as np
import scipy
from scipy import linalg
import time

Transition = namedtuple('Transition',
						('state', 'action','reward', 'next_state', 'done'))

class LSPIAgent(object):
	"""docstring for LSPIAgent"""
	def __init__(self, param):
		super(LSPIAgent, self).__init__()
		self.state_dim = param['state_dim']
		self.n_actions = param['n_actions']
		self.basisfunc_dim = param['basis_function_dim']
		self.gamma = param['weight_discount']
		self.stop_criterion = param['stop_criterion']
		
		self.basis_function = RBF(self.state_dim, self.basisfunc_dim, self.n_actions, self.gamma)
		self.n_basis_func = self.basis_function.size()
		epsilon = 1-param['exploration']
		self.policy = GreedyPolicy(self.basis_function, self.n_actions, epsilon)
		# print(self.policy.weights)
		self.lstdq = LSTDQ(self.basis_function, self.gamma)
		self.n_iter_max = 30

	def train(self, sample):
		error = float('inf')
		error_his = []
		i_iter = 0
		while error > self.stop_criterion and i_iter<self.n_iter_max: 
			new_weights = self.lstdq.training(sample, self.policy)
			error = np.linalg.norm((new_weights - self.policy.weights))
			print("error when update_weights in interation {}: {}".format(i_iter,error))
			error_his.append(error)
			self.policy.update_weights(new_weights)
			i_iter += 1
		# time.sleep(2)
		# print(error_his)
		# print(self.policy.weights)
		return error_his


	def get_action(self, state):
		return self.policy.get_best_action_epsilon(state)


class LSTDQ(object):
	"""docstring for LSTDQ"""
	def __init__(self, basis_function, gamma):
		super(LSTDQ, self).__init__()
		self.basis_function = basis_function
		self.gamma = gamma

	def training(self, samples, policy):
		# Figure 5 in paper
		n = self.basis_function.size()
		A = np.zeros([n, n])
		b = np.zeros([n, 1])
		# l2 weights here:
		np.fill_diagonal(A, .1)  # Singular matrix error

		for s in samples:
			state = s.state
			action = s.action
			reward = s.reward
			next_state = s.next_state
			done = s.done
			phi_sa = (policy.basis_func.evaluate(state, action)
					  .reshape((-1, 1)))

			if not done:
				best_action = policy.get_best_action(next_state)
				phi_sprime = (policy.basis_func
							  .evaluate(next_state, best_action)
							  .reshape((-1, 1)))
			else:
				phi_sprime = np.zeros((n, 1))

			A += phi_sa.dot((phi_sa - self.gamma*phi_sprime).T)
			b += phi_sa*reward

		a_rank = np.linalg.matrix_rank(A)
		if a_rank == n:
			w = linalg.solve(A, b)
		else:
			logging.warning('A matrix is not full rank. %d < %d', a_rank, n)
			w = linalg.lstsq(A, b)[0]
		return w.reshape((-1, ))

		# 	phi = self.basis_function.evaluate(state, action)
		# 	if not done:
		# 		_action = policy.get_best_action_epsilon(state)
		# 		phi_next = self.basis_function.evaluate(next_state, _action)
		# 	else:
		# 		phi_next = np.zeros(n)
		# 	loss = phi - self.gamma*phi_next
		# 	phi = np.reshape(phi, [n, 1])
		# 	loss = np.reshape(loss, [1, n])
		# 	A += np.dot(phi, loss)
		# 	b += phi * reward
		# inv_A = np.linalg.inv(A)
		# w = np.dot(inv_A, b)
		# return w

