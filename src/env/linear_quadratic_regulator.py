import numpy as np
from .spaces.box import *
from .spaces.discrete import *
import random
from numpy import matlib as mb
import scipy.linalg

class LQREnv(object):
	"""docstring for LQREnv"""
	def __init__(self, A=np.matrix(0.9), 
					   B=np.matrix(1.), 
					   Z1=np.matrix(1.), 
					   Z2=np.matrix(1.), 
					   noise_cov=np.matrix(1.), 
					   seed=1):
		'''
		A,B are some Matrices here
			if state is mx1 then A is mxm, B is mxn then the u should be nx1
		Z1,Z2 are some positive semi-definite weight matrices mxm or scala
			that determines the trade-off between keeping state small and keeping action small.
		'''
		super(LQREnv, self).__init__()
		self.m = A.shape[0]
		assert B.shape[0] == A.shape[1]
		self.n = B.shape[1]
		if not np.isscalar(Z1):
			Z1.shape[0] == A.shape[0]
		if not np.isscalar(Z2):
			Z2.shape == A.shape

		self.A = A
		self.B = B
		self.Z1 = Z1
		self.Z2 = Z2
		self.state = None
		# noisy
		self.noise_mu = np.matrix(np.zeros(self.m).reshape(self.m,1))
		self.noise_cov = noise_cov
		self.rng = self.set_seed(seed)

		high = mb.ones((self.n,1))*np.inf
		# TODO: Action now is continous
		self.action_space = Box(low=-high, high=high, dtype=np.float32)
		high = mb.ones((self.m,1))*np.inf
		# print(self.action_space)
		self.observation_space = Box(low=-high, high=high, dtype=np.float32)


	def set_seed(self, seed):
		if seed<=0:
			raise ValueError('Seed must be a non-negative integer not {}'.format(seed))
		rng = np.random.RandomState()
		rng.seed(seed)
		return rng


	def calcu_reward(self, u):
		x = self.state
		# cost matrix size 1x1
		# u = np.matrix(u)
		
		cost = x.T*self.Z1*x + u.T*self.Z2*u
		return -float(cost)


	def step(self, action):
		reward = self.calcu_reward(action)
		noise = self.rng.normal(self.noise_mu, self.noise_cov, self.m).reshape(self.m,1)
		# change to matrix 
		# print("action: {}".format(action))
		new_state = self.A*self.state + self.B*action + noise
		# print("new_state: {}".format(new_state))
		self.state = new_state
		return self._get_obs(), reward, False, {}


	def reset(self):
		self.state = mb.rand(self.m, 1)
		return self.state


	def _get_obs(self):
		# noise = self.rng.normal(self.noise_mu, self.noise_cov, 1)[0]
		return self.state


	def close(self):
		self.state = None

	# SCALA
	def true_weights_scala(self, L, gamma):
		# state and action np.matrix mx1
		Z1 = self.Z1
		Z2 = self.Z2
		A = self.A
		B = self.B

		q = Z1 + L.T*Z2*L 
		a = np.sqrt(gamma)*(A-B*L)
		
		P = scipy.linalg.solve_discrete_lyapunov(a, q)

		w1 = gamma/(1-gamma)*np.trace(P)
		w2 = (Z1 + gamma*A.T*P*A).item()
		w3 = (Z2 + gamma*B.T*P*B).item()
		w4 = (2*gamma*A.T*P*B).item()

		return np.matrix([-w1,-w2,-w3,-w4]).reshape(4,1)

	def true_Qvalue(self, L, gamma, state, action):
		x = state
		u = action
		Z1 = self.Z1
		Z2 = self.Z2
		A = self.A
		B = self.B

		q = Z1 + L.T*Z2*L
		a = np.sqrt(gamma)*(A-B*L).T
		
		P = scipy.linalg.solve_discrete_lyapunov(a, q)

		w1 = gamma/(1-gamma)*np.trace(self.noise_cov*P)
		w2 = Z1 + gamma*A.T*P*A
		w3 = Z2 + gamma*B.T*P*B
		w4 = 2*gamma*A.T*P*B
		q_value = -(x.T*w2*x + u.T*w3*u + x.T*w4*u + w1).item()
		return q_value

	def true_Qvalue_state(self, L, gamma, state):
		x = state
		Z1 = self.Z1
		Z2 = self.Z2
		A = self.A
		B = self.B

		q = Z1 + L.T*Z2*L
		a = np.sqrt(gamma)*(A-B*L).T
		
		P = scipy.linalg.solve_discrete_lyapunov(a, q)

		q_state = -(x.T*P*x + gamma/(1-gamma)*np.trace(self.noise_cov*P))
		return q_state.item()

	def optimal_policy_L(self, gamma):
		Z1 = self.Z1
		Z2 = self.Z2
		A = self.A
		B = self.B

		a = np.sqrt(gamma)*A
		b = np.sqrt(gamma)*B
		r = Z2
		q = Z1
		P = scipy.linalg.solve_discrete_are(a,b,q,r)

		L = gamma* (1/(gamma*B.T*P*B+Z2))*B.T*P*A
		return L
		

if __name__ == '__main__':
	# test env for 2d
	A = np.matrix([[0.5,1],[0,1]])
	B = np.matrix([[0],[1]])
	Z1 = np.matrix([[1,0],[0,0]])
	Z2 = 0.1
	noise_cov = np.matrix([[1,0],[0,1]])
	env = LQREnv(A=A,B=B,Z1=Z1,Z2=Z2,noise_cov=noise_cov)
	state = env.reset()
	action = np.matrix(mb.rand(env.n,env.n))
	env.step(action)
	L = mb.rand(env.n, env.m)
	env.true_Qvalue( L, gamma, x,u)













