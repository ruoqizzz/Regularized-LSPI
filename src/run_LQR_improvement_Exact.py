# -*- coding: utf-8 -*-

import argparse
from lspi import LSPIAgent
from replay_buffer import ReplayBuffer
import gym
from env.linear_quadratic_regulator import LQREnv
from basis_func import *
import time
from collect_samples import *
from policy import *
import matplotlib.pyplot as plt
import numpy as np
import time

# sample data files name for LQR
LQR_samples_filename = {
	2000: "samples/LQR/gaussian_actions_2000.pickle",
	5000: "samples/LQR/gaussian_actions_5000.pickle"
}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="LQR", choices=["cliff-v0","CartPole-v0","inverted_pedulum","LQR","chain"])	# gym env to train
	parser.add_argument('--episode_num', default=10, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=10, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_steps', default="5000", choices=["2000","5000","10000","20000"])
	parser.add_argument('--max_steps', default=500, type=int)
	parser.add_argument('--batch_size', default=2000, type=int)
	parser.add_argument('--update_freq', default=10000000, type=int)
	parser.add_argument('--L', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--reg_opt', default="none", choices=["l1","l2", "wl1", "none"])
	parser.add_argument('--reg_param', default=0.01, type=float)
	
	args = parser.parse_args()
	params = vars(args)

	# env
	env = LQREnv()
	params['n_actions'] = env.action_space.shape[0]
	params['state_dim'] = env.observation_space.shape[0]
	params['basis_func'] = ExactBasis4LQR()
	params['sample_max_steps'] = int(params['sample_max_steps'])
	gamma = params['weight_discount']
	# Note: now init policy with specific L
	#		the action would be related to this init L
	#		Remember to update L!
	L=np.matrix(params['L'])
	params['policy'] = ExactPolicy4LQR(params['basis_func'],L)

	# set the parameters for agent
	batch_size = params['batch_size']
	update_freq = params['update_freq']
	n_episode = params['episode_num']
	max_steps = params['max_steps']

	agent = LSPIAgent(params)
	
	sample_filename = LQR_samples_filename[params['sample_max_steps']]
	f = open(sample_filename, 'rb')
	replay_buffer = pickle.load(f)
	# training to get weights -> best L 
	sample = replay_buffer.sample(batch_size)
	error_list, new_weights = agent.train(sample)

	# log
	reward_his = []
	estimateL_his = []
	i_update = 0
	for i_episode in range(n_episode):
		state = env.reset()
		i_episode_steps = 0
		accu_reward = 0
		# LQR never done
		# print("i_episode: {}".format(i_episode))
		while True:
			i_episode_steps+=1
			action = agent.get_action(state)
			state_, reward, done, info = env.step(action[0])
			# print("state: {}".format(state))
			# print("action: {}".format(action))
			# print("reward: {}".format(reward))
			# print("state_: {}\n".format(state_))
			# replay_buffer.store(state, action, reward, state_, done)
			accu_reward += reward
			state = state_
			if i_episode_steps>20:
				# done
				# print("accu_reward {}\n".format(accu_reward))
				reward_his.append(accu_reward)
				time.sleep(0.1)
				break
		# estimateL = agent.policy.estimate_policy_L().item()
		# use true Q/weights in this L to check whether it converge to optimal one 
		true_weights = env.true_weights_scala(agent.policy.L, gamma)
		w3 = true_weights[2].item()
		w4 = true_weights[3].item()
		estimateL = np.matrix(w4/(2*w3))
		estimateL_his.append(estimateL.item())
		agent.policy.L = estimateL
		print("estimateL: {}".format(estimateL))
		agent.train(sample)
			
	# now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
	trueL = env.optimal_policy_L(gamma).item()
	print("trueL: {}".format(trueL))
	print("estimateL_his: {}",estimateL_his)
	env.close()
	replay_buffer.reset()

	# plot 
	# plt.plot(reward_his)
	# plt.show()
	plt.plot(np.arange(n_episode), estimateL_his, label='estimate L')
	plt.plot(np.arange(n_episode), [trueL]*n_episode, label='optimal L')
	plt.ylabel('L')
	plt.xlabel('iteration')
	plt.legend(loc='upper right')
	plt.show()

if __name__ == '__main__':
	main()







