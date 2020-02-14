import numpy as np
from .spaces.box import *
from .spaces.discrete import *

class LQREnv(object):
	"""docstring for LQREnv"""
	def __init__(self, A=0.9, B=1., Z1=1., Z2=1., noise_cov = 1., seed=1):
		'''
		A,B are some Matrices here
		Z1,Z2 are some positive semi-definite weight matrices
			that determines the trade-off between keeping state small and keeping action small.
		'''
		super(LQREnv, self).__init__()
		self.A = A
		self.B = B
		self.Z1 = Z1
		self.Z2 = Z2
		self.state = None
		# noisy
		self.noise_mu = 0.
		self.noise_cov = noise_cov
		self.rng = self.set_seed(seed)

		high = np.array([1.])
		# TODO: Action now is continous
		self.action_space = Box(low=-high, high=high, dtype=np.float32)
		self.observation_space = Box(low=-high, high=high, dtype=np.float32)



	def set_seed(self, seed):
		if seed<=0:
			raise ValueError('Seed must be a non-negative integer not {}'.format(seed))
		rng = np.random.RandomState()
		rng.seed(seed)
		return rng


	def calcu_cost(self, action):
		x = np.array(self.state)
		u = np.array(self.u)
		return np.dot(x.T,x) + np.dot(u.T,u)


	def step(self, action):
		cost = self.calcu_cost(action)
		return self._get_obs(), -cost, False, {}


	def reset(self):
		self.state = None


	def _get_obs(self):
		noise = self.rng.normal(self.noise_mu, self.noise_cov, 1)[0]
		return self.state + noise


	def close(self):
		self.state = None
