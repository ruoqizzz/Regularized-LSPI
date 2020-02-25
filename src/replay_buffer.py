# -*- coding: utf-8 -*-
import random
from collections import namedtuple

MAX_BUFFSIZE = 100000

Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))

class ReplayBuffer(object):
	"""docstring for ReplayBuffer"""
	def __init__(self, buffer_size=MAX_BUFFSIZE, seed=1):
		super(ReplayBuffer, self).__init__()
		self.buffer_size = buffer_size
		self.random_seed = random.seed(seed)
		self.buffer = []
		self.next_index = 0
		self.num_buffer = 0


	def store(self, *transition):
		# store transition
		if self.num_buffer < self.buffer_size:
			self.buffer.append(None)
		self.buffer[self.next_index] = Transition(*transition)
		self.next_index = (self.next_index+1)%self.buffer_size
		self.num_buffer += 1
		# print(self.num_buffer)

	def sample(self, batch_size):
		if batch_size < self.num_buffer:
			# rand =  random.sample(self.buffer[:int(self.num_buffer-batch_size/2)], int(batch_size/2))
			# concat = rand + self.buffer[-int(batch_size/2):]
			# return [-batch_size:] to ensure the number is correct and contains the latest sample
			# return concat[-batch_size:]
			return random.sample(self.buffer, batch_size)
		else:
			return random.sample(self.buffer, self.num_buffer)


	def sample_with_current(self, batch_size, current_size=0):
		assert batch_size >= current_size
		if batch_size < self.num_buffer:
			if current_size==0:
				return random.sample(self.buffer, batch_size)
			elif batch_size==current_size:
				return self.buffer[-current_size:]
			else: 
				rand =  random.sample(self.buffer[:int(self.num_buffer-current_size)], int(batch_size-current_size))
				concat = rand + self.buffer[-current_size:]
				return concat[-batch_size:]
		else:
			return random.sample(self.buffer, self.num_buffer)


	def reset(self):
		self.buffer = []
		self.next_index = 0
		self.num_buffer = 0
