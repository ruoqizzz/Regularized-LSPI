# -*- coding: utf-8 -*-
from replay_buffer import ReplayBuffer
from collections import namedtuple
import numpy as np
Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))

def collect_samples(env, agent, max_episodes, max_steps, option):
	ifRandom = False
	replay_buffer = ReplayBuffer()
	if agent==None:
		# random policy
		ifRandom = True
	i_episode = 0
	while i_episode<max_episodes:
		state = env.reset()
		done  = False
		i_episode_steps = 0
		while True and i_episode_steps<max_steps:
			i_episode_steps += 1
			if ifRandom:
				action = env.action_space.sample()
			else:
				action = agent.get_action(state)
			state_, reward, done, info = env.step(action)
			replay_buffer.store(state, action, reward, state_, done)
			if done:
				break
	return replay_buffer


def collect_samples_gaussian(env, max_steps):
	replay_buffer = ReplayBuffer()
	i_episode_steps = 0
	state = env.reset()
	done  = False
	while True and i_episode_steps<max_steps:
		i_episode_steps += 1
		# action = env.action_space.sample()
		action = np.matrix(np.random.normal(0, 1, 1)[0])
		state_, reward, done, info = env.step(action)
		replay_buffer.store(state, action, reward, state_, done)
		if done:
			break
	return replay_buffer


