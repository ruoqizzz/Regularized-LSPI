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
	parser.add_argument('--basis_function_dim', default=20, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--batch_size', default=1000, type=int)
	parser.add_argument('--update_freq', default=1000, type=int)
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2"])
	parser.add_argument('--reg_param', default=0.001, type=float)
	parser.add_argument('--rbf_sigma', default=0.5, type=float)
	
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
	samples = []
	# this replay_buffer already with samples
	# fns = ["samples/CartPole/CartPole-500","samples/CartPole/CartPole-1000"]
	fns = [
	"samples/CartPole/CartPole-100",
	"samples/CartPole/CartPole-200",
	"samples/CartPole/CartPole-300",
	"samples/CartPole/CartPole-400",
	"samples/CartPole/CartPole-500",
	"samples/CartPole/CartPole-600",
	"samples/CartPole/CartPole-700",
	"samples/CartPole/CartPole-800",
	"samples/CartPole/CartPole-900",
	"samples/CartPole/CartPole-1000"]

	for fn in fns:
		f = open(fn, 'rb')
		replay_buffer = pickle.load(f)
		samples.append(replay_buffer.buffer)
		f.close()

	sample_meanmean = []
	sample_meanmax = []
	sample_meanmin = []
	for i_samples in samples:
		test_mean = []
		test_max = []
		test_min = []
		for i_test in range(10):
			for i in range(20):
				print("agent training {} times".format(i))
				agent.train(i_samples)
				print("\n")
			# evalute the policy after 20 policy iteration
			reward_history = []
			steps_history = []
			for i in range(1000):
				state = env.reset()
				done  = False
				total_reward = 0
				total_steps = 0
				i_episode_steps = 0
				while True:
					# env.render()
					action = agent.get_action(state)
					state_, steps, done, info = env.step(action)
					if params['env_name']=="CartPole-v0":
						# recalculate reward
						x, x_dot, theta, theta_dot = state_
						r1 = -x**2
						r2 = - 10*theta**2
						reward = r1+r2
					state = state_
					total_reward += reward
					total_steps += steps
					if done:
						reward_history.append(total_reward)
						steps_history.append(total_steps)
						# print("i_episode_steps {}".format(i_episode_steps))
						print("total_reward {}".format(total_reward))
						print("total_steps {}".format(total_steps))
						# time.sleep(0.1)
						break
		test_mean.append(np.mean(steps_history))
		print("history mean {}".format(np.mean(steps_history)))
		test_max.append(np.max(steps_history))
		print("history max {}".format(np.max(steps_history)))
		test_min.append(np.min(steps_history))
		print("history min {}".format(np.min(history)))

		sample_meanmean.append(np.mean(test_mean))		
		sample_meanmax.append(np.mean(test_max))
		sample_meanmin.append(np.mean(test_min))	
	env.close()
	replay_buffer.reset()
	# plot
	plt.plot(sample_meanmean)
	plt.plot(sample_meanmax)
	plt.plot(sample_meanmin)
	plt.show()


if __name__ == '__main__':
	main()
