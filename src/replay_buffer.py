# -*- coding: utf-8 -*-
import random
from collections import namedtuple
import pandas as pd
import numpy as np

MAX_BUFFSIZE = 100000

Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))

class ReplayBuffer(object):
	"""docstring for ReplayBuffer"""
	def __init__(self, buffer_size=MAX_BUFFSIZE, seed=1):
		super(ReplayBuffer, self).__init__()
		self.buffer_size = buffer_size
		self.random_seed = random.seed(seed)
		self.buffer = pd.DataFrame(columns=['state', 'action','reward', 'next_state', 'done'])
		self.next_index = 0
		self.num_buffer = 0


	def store(self, *transition):
		# store transition
		self.buffer.loc[self.next_index] = Transition(*transition)._asdict()
		self.next_index = (self.next_index+1)%self.buffer_size
		self.num_buffer = min(self.buffer_size, self.num_buffer+1)

	def sample(self, batch_size):
		if batch_size <= self.num_buffer:
			samples = self.buffer.sample(n=batch_size)
			return np.array(samples['state'].to_list()), samples['action'].to_list(), np.array(samples['reward'].to_list()), np.array(samples['next_state'].to_list()), np.array(samples['done'].to_list())
		else:
			return np.array(self.buffer['state'].to_list()), self.buffer['action'].to_list(), np.array(self.buffer['reward'].to_list()), np.array(self.buffer['next_state'].to_list()), np.array(self.buffer['done'].to_list())


	def reset(self):
		self.buffer = pd.DataFrame(columns=['state', 'action','reward', 'next_state', 'done'])
		self.next_index = 0
