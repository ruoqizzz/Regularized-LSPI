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
import pickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="CartPole-v0", choices=["cliff-v0","CartPole-v0","inverted_pedulum","chain"])	# gym env to train
	parser.add_argument('--episode_num', default=1000, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=200, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--batch_size', default=1000, type=int)
	parser.add_argument('--update_freq', default=1000, type=int)
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2"])
	parser.add_argument('--reg_param', default=0.0001, type=float)
	parser.add_argument('--rbf_sigma', default=0.5, type=float)
	
	args = parser.parse_args()
	params = vars(args)

	env = gym.make(params['env_name'])

	# set the parameters for agent
	batch_size = params['batch_size']
	update_freq = params['update_freq']
	n_episode = params['episode_num']
	gamma = params['weight_discount']
	n_features = params['basis_function_dim']


	params['n_actions'] = env.action_space.n
	params['state_dim'] = env.observation_space.shape[0]


	# replay buffers
	rbs = []
	# this replay_buffer already with samples
	for i in np.arange(1,11)*100:
		fn = "samples/CartPole/reward_shape/CartPole"+str(i)+".pickle"
		f = open(fn, 'rb')
		replay_buffer = pickle.load(f)
		rbs.append(replay_buffer)
		f.close()
	# the number of samples's episodes
	num = np.arange(1,11)*100
	sample_meanmean = []
	sample_meanmax = []
	sample_meanmin = []
	for i_s in range(len(rbs)):
		print("===================================")
		print("sample {}\n".format(num[i_s]))
		i_samples = rbs[i_s].sample(rbs[i_s].num_buffer)
		test_mean = []
		test_max = []
		test_min = []
		for i_test in range(10):
			print("\ntest "+str(i_test))
			# reset 
			basis_func = RBF(params['state_dim']-2, n_features, params['n_actions'], params['rbf_sigma'], high=np.array([0.21,2.7]))
			params['basis_func'] = basis_func
			policy = GreedyPolicy(params['basis_func'], params['n_actions'], 1-params['exploration'])
			params['policy'] = policy
			agent = LSPIAgent(params)
			for i in range(1):
				print("agent training {} times".format(i))
				agent.train(i_samples)
				print("")
			# evalute the policy after 20 policy iteration
			history = []
			for i in range(1000):
				state = env.reset()
				done  = False
				total_reward = 0
				i_episode_steps = 0
				while True:
					# env.render()
					state = np.reshape(state[2:4], (1,2))
					action = agent.get_action(state)
					# print("action: {}".format(action))
					state_, reward, done, info = env.step(action[0])
					# if params['env_name']=="CartPole-v0":
					# 	# recalculate reward
					# 	x, x_dot, theta, theta_dot = state_
					# 	r1 = -(10*x)**2
					# 	r2 = - (10*theta)**2
					# 	reward = r1+r2
					state = state_
					total_reward += reward
					if done:
						history.append(total_reward)
						# print("i_episode_steps {}".format(i_episode_steps))
						# print("total_reward {}".format(total_reward))
						# time.sleep(0.1)
						break
			test_mean.append(np.mean(history))
			print("history mean {}".format(np.mean(history)))
			test_max.append(np.max(history))
			print("history max {}".format(np.max(history)))
			test_min.append(np.min(history))
			print("history min {}".format(np.min(history)))
		sample_meanmean.append(np.mean(test_mean))
		print("\ntest_mean {}\n".format(np.mean(test_mean)))
		sample_meanmax.append(np.mean(test_max))
		sample_meanmin.append(np.mean(test_min))	


	print(sample_meanmean)
	print(sample_meanmax)
	print(sample_meanmin)

	plt.plot(num,sample_meanmean)
	plt.plot(num,sample_meanmax)
	plt.plot(num,sample_meanmin)
	plt.ylabel('Reward')
	plt.xlabel('Episode')
	plt.show()




if __name__ == '__main__':
	main()
