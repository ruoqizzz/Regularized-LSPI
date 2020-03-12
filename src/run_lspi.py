# -*- coding: utf-8 -*-
import argparse
from lspi import LSPIAgent
from replay_buffer import ReplayBuffer
import gym
import gymgrid
from env.inverted_pendulum import InvertedPendulumEnv
from env.linear_quadratic_regulator import LQREnv
from env.chain import ChainEnv
from basis_func import *
import time
from collect_samples import *
from policy import *
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="CartPole-v0", choices=["cliff-v0","CartPole-v0","inverted_pedulum","chain"])	# gym env to train
	parser.add_argument('--episode_num', default=2000, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=200, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--batch_size', default=1000, type=int)
	parser.add_argument('--update_freq', default=1000, type=int)
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2"])
	parser.add_argument('--reg_param', default=0.001, type=float)
	parser.add_argument('--rbf_sigma', default=0.001, type=float)
	
	args = parser.parse_args()
	params = vars(args)

	# set env
	if params['env_name']=="inverted_pedulum":
		env = InvertedPendulumEnv()
	elif params['env_name']=="chain":
		env = ChainEnv()
	else:
		env = gym.make(params['env_name'])

	# set the parameters for agent
	batch_size = params['batch_size']
	update_freq = params['update_freq']
	n_episode = params['episode_num']

	params['n_actions'] = env.action_space.n
	params['state_dim'] = env.observation_space.shape[0]

	basis_func = RBF(params['state_dim'], params['basis_function_dim'], params['n_actions'], params['rbf_sigma'])
	params['basis_func'] = basis_func
	params['policy'] = GreedyPolicy(basis_func, params['n_actions'], 1-params['exploration'])
	agent = LSPIAgent(params)
	# this replay_buffer already with samples
	replay_buffer = ReplayBuffer()
	history = []
	total_steps = 0 
	i_update = 0

	for i_episode in range(n_episode):
		state = env.reset()
		done  = False
		total_reward = 0
		i_episode_steps = 0
		while True:
			i_episode_steps += 1
			total_steps += 1
			# env.render()
			action = agent.get_action(state)
			state_, reward, done, info = env.step(action)
			# if params['env_name']=="CartPole-v0":
			# 	# recalculate reward
			# 	x, x_dot, theta, theta_dot = state_
			# 	r1 = -(10*x)**2
			# 	r2 = - (10*theta)**2
			# 	reward = r1+r2
			replay_buffer.store(state, action, reward, state_, done)
			total_reward += reward
			if total_steps%update_freq==0:
				i_update += 1
				print("i_update {}".format(i_update))
				sample = replay_buffer.buffer[-batch_size:]
				error_list, new_weights = agent.train(sample)
			if done:
				# print("i_episode_steps {}".format(i_episode_steps))
				print("total_reward {}".format(total_reward))
				history.append(i_episode_steps)
				# time.sleep(0.1)
				break

	env.close()
	replay_buffer.reset()
	# plot
	plt.plot(history)


if __name__ == '__main__':
	main()
