# -*- coding: utf-8 -*-
from replay_buffer import ReplayBuffer
from collections import namedtuple
import numpy as np
import pickle
from env.linear_quadratic_regulator import LQREnv

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
			state = state_
			if done:
				break
	return replay_buffer

# collect samples using gaussain actions
def collect_samples_gaussian(env, max_steps):
	replay_buffer = ReplayBuffer()
	i_episode_steps = 0
	state = env.reset()
	done  = False
	while i_episode_steps<max_steps:
		i_episode_steps += 1
		# action = env.action_space.sample()
		action = np.matrix(np.random.normal(0, 1, 1)[0])
		state_, reward, done, info = env.step(action)
		replay_buffer.store(state, action, reward, state_, done)
		state = state_
	return replay_buffer
	# f = open(filename, 'wb')
	# pickle.dump(replay_buffer, f)

def collect_LQR_gaussian(env, max_steps):
	# shape of state
	m = env.m
	# shape of action
	n = env.n
	replay_buffer = ReplayBuffer()
	i_episode_steps = 0
	state = env.reset()
	done  = False
	while i_episode_steps < max_steps:
		i_episode_steps += 1
		action = np.matrix(np.random.normal(0, 1, n).reshape(n,1))
		state_, reward, done, info = env.step(action)
		replay_buffer.store(state, action, reward, state_, done)
		state = state_
	return replay_buffer

def collect_samples_sa(env, states,L):
	replay_buffer = ReplayBuffer()
	env.reset()
	for i in range(len(states)):
		state = np.matrix(states[i])
		action = -L*state
		# print("action: {}".format(action.T))
		state_, reward, done, info = env.step(action)
		replay_buffer.store(state, action, reward, state_, done)
		
	return replay_buffer

if __name__ == '__main__':
	# collect samples for 2000 steps
	env = LQREnv()
	# sample_step_list = [10000, 20000]
	# for steps in sample_step_list:
	# 	replay_buffer = collect_samples_gaussian(env, steps)
	# 	f1 = open("samples/LQR/gaussian_actions_"+str(steps)+".pickle", 'wb')
	# 	pickle.dump(replay_buffer, f1)
	# 	f1.close()

	states = np.linspace(-2.0, 2.0, 10000)
	L = np.matrix(0.1)
	replay_buffer = collect_samples_sa(env, states, L)
	f1 = open("samples/LQR/states[-2,2]_10000_L=0.1.pickle", 'wb')
	pickle.dump(replay_buffer, f1)
	f1.close()



