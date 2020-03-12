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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="CartPole-v0", choices=["cliff-v0","CartPole-v0","inverted_pedulum","chain"])	# gym env to train
	parser.add_argument('--episode_num', default=10000, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=10, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_episodes', default=300, type=int)
	parser.add_argument('--sample_max_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=1000, type=int)
	parser.add_argument('--update_freq', default=500, type=int)
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

	basis_func = RBF(params['state_dim'], params['basis_function_dim'], params['n_actions'], params['weight_discount'])
	params['basis_func'] = basis_func
	params['policy'] = GreedyPolicy(basis_func, params['n_actions'], 1-params['exploration'])
	agent = LSPIAgent(params)
	# this replay_buffer already with samples
	replay_buffer = collect_samples(env, agent, params['sample_max_episodes'], params['sample_max_steps'])
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
			# if total_steps > params['learning_start']:
			# 	env.render()
			# env.render()
			action = agent.get_action(state)
			state_, reward, done, info = env.step(action)
			if params['env_name']=="CartPole-v0":
				# recalculate reward
				x, x_dot, theta, theta_dot = state_
				r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
				r2 = - abs(theta) * abs(theta)/env.theta_threshold_radians
				reward = r1+r2
			replay_buffer.store(state, action, reward, state_, done)
			total_reward += reward
			if total_steps%update_freq==0:
				i_update += 1
				print("i_update {}".format(i_update))
				sample = replay_buffer.sample(batch_size)
				error_list, new_weights = agent.train(sample)
			if done:
				# print("i_episode_steps {}".format(i_episode_steps))
				print("total_reward {}".format(total_reward))
				history.append(total_reward)
				# time.sleep(0.1)
				break

	env.close()
	replay_buffer.reset()



if __name__ == '__main__':
	main()
