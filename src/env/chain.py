import numpy as np
from .spaces.box import *
from .spaces.discrete import *
import random 

class ChainEnv(object):
	"""docstring for ChainEnv"""
	def __init__(self, n_states=4, failure_probability=0.1):
		super(ChainEnv, self).__init__()
		self.n_states = n_states
		# always 2 : [0,1] = ['left', 'right']
		self.n_actions = 2
		self.failure_prob = failure_probability

		self.action_space = Discrete(self.n_actions)
		self.observation_space = Discrete(self.n_states)
		self.state = None
		self.states = list(range(self.n_states))
		

	def set_seed(self, seed):
		if seed<=0:
			raise ValueError('Seed must be a non-negative integer not {}'.format(seed))
		rng = np.random.RandomState()
		rng.seed(seed)
		return rng


	def step(self, action):
		if action not in [0,1]:
			raise ValueError("Action is not avaible")
		action_failed = False
		if random.random() < self.failure_prob:
			action_failed = True
		# the two boundaries of the chain are dead-ends. 
		# The reward vector over states is (0, +1, +1, 0) and the discount factor is set to 0.9.
		# assume the reward vector is in the middle
		if action==0 or (action==1 and action_failed):
			state_ = max(0, self.state-1)
		else:
			state_ = min(self.n_states-1, self.state+1)

		reward = 0
		done = False
		if state_==0 or state_==self.n_states-1:
			done = True
		elif state_ == int(self.n_states/2) or state_ == int(self.n_states/2 + 1):
			reward = 1
		self.state = state_
		return self._get_obs(), reward, done, {}


	def reset(self):
		# print("self.states: {}".format(self.states))
		# sample from state not done
		self.state = random.sample(self.states[1:-1], 1)[0]
		return self._get_obs()

	def _get_obs(self):
		# noise = self.rng.normal(self.noise_mu, self.noise_cov, 1)[0]
		return self.state


	def close(self):
		self.state = None
		