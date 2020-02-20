# -*- coding: utf-8 -*-
import argparse
from lspi import LSPIAgent
from replay_buffer import ReplayBuffer
import gym
import gymgrid
from env.linear_quadratic_regulator import LQREnv
from basis_func import *
import time
from collect_samples import *
from policy import *
import matplotlib.pyplot as plt

def main():

	parser = argparse.ArgumentParser()
# -*- coding: utf-8 -*-
	parser.add_argument('--env_name', default="LQR", choices=["cliff-v0","CartPole-v0","inverted_pedulum","LQR","chain"])	# gym env to train
	parser.add_argument('--episode_num', default=300, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=10, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_episodes', default=300, type=int)
	parser.add_argument('--sample_max_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=5000, type=int)
	parser.add_argument('--update_freq', default=2500, type=int)
	args = parser.parse_args()
	params = vars(args)

	env = LQREnv()
	params['n_actions'] = env.action_space.shape[0]
	params['state_dim'] = env.observation_space.shape[0]
	params['basis_func'] = ExactBasis4LQR()
	params['policy'] = ExactPolicy4LQR(params['basis_func'], np.matrix(0.3), 1-params['exploration'])

	# set the parameters for agent
	batch_size = params['batch_size']
	update_freq = params['update_freq']
	n_episode = params['episode_num']


	agent = LSPIAgent(params)
	print('collecting samples....')
	replay_buffer = collect_samples_gaussian(env, 2500)
	print('collecting done, buffer size: {}'.format(replay_buffer.num_buffer))

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
			action = agent.policy.get_best_action(state)
			state_, reward, done, info = env.step(action)
			replay_buffer.store(state, action, reward, state_, done)
			total_reward += reward
			if total_steps%update_freq==0:
				i_update += 1
				print("i_update {}".format(i_update))
				sample = replay_buffer.sample(batch_size)
				error_list = agent.train(sample)
			if i_episode_steps == params['sample_max_steps']:
				done = True
			if done:
				# print("i_episode_steps {}".format(i_episode_steps))
				print("total_reward {}".format(total_reward))
				history.append(total_reward)
				# time.sleep(0.1)
				break

	env.close()
	replay_buffer.reset()
	plt.plot(history)
	plt.ylabel('reward')
	plt.show()



if __name__ == '__main__':
	main()







