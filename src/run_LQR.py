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
import numpy as np
import time

def main():

	parser = argparse.ArgumentParser()
# -*- coding: utf-8 -*-
	parser.add_argument('--env_name', default="LQR", choices=["cliff-v0","CartPole-v0","inverted_pedulum","LQR","chain"])	# gym env to train
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=10, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_steps', default=2000, type=int)
	parser.add_argument('--max_steps', default=500, type=int)
	parser.add_argument('--batch_size', default=2000, type=int)
	parser.add_argument('--update_freq', default=10000000, type=int)
	args = parser.parse_args()
	params = vars(args)

	env = LQREnv()
	params['n_actions'] = env.action_space.shape[0]
	params['state_dim'] = env.observation_space.shape[0]
	params['basis_func'] = ExactBasis4LQR()




	L=np.matrix(0.1)
	gamma = params['weight_discount']
	params['policy'] = ExactPolicy4LQR(params['basis_func'], L, 1-params['exploration'])

	# set the parameters for agent
	batch_size = params['batch_size']
	update_freq = params['update_freq']
	max_steps = params['max_steps']

	agent = LSPIAgent(params)
	print('collecting samples....')
	replay_buffer = collect_samples_gaussian(env, 2500)
	print('collecting done, buffer size: {}'.format(replay_buffer.num_buffer))

	true_weights_his = []
	true_estimate_error_history = []
	q_true_his = []
	q_estimate_his = []

	sample = replay_buffer.sample(batch_size)
	error_list, new_weights = agent.train(sample)




	state = np.matrix(-1.)
	actions = np.linspace(-1,1, 100)

	for i in range(len(actions)):
		action = np.matrix(actions[i])
		q_estimate = (agent.policy.q_state_action_func(state, action)).item()
		# print("q_estimate: {}".format(q_estimate))
		q_true = env.true_Qvalue(L, gamma, state, action)
		# print("q_true: {}".format(q_true))
		q_estimate_his.append(q_estimate)
		q_true_his.append(q_true)

	true_weights = env.true_weights(L, gamma)
	print("true_weights: {}".format(true_weights))
	estimate_weights = agent.policy.weights
	print("estimate_weights: {}".format(estimate_weights))
	true_estimate_error = np.linalg.norm(true_weights-estimate_weights)
	print("true_estimate_error: {}".format(true_estimate_error))

	now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

	plt.figure(figsize=(8, 6))
	plt.subplot(211)
	plt.plot(actions, q_estimate_his)
	plt.title('q estimate')

	plt.subplot(212)
	plt.plot(actions, q_true_his)
	plt.title('q true')
	plt.savefig(now+"q_true&estimate-action(-1,1)")
	plt.show()


	states = np.linspace(-2.0, 2.0, 100)
	actions = -L.item()*states

	for i in range(len(states)):
		state = states[i]
		action = actions[i]
		q_estimate = (agent.policy.q_state_action_func(state, action)).item()
		q_true = env.true_Qvalue(L, gamma, state, action)
		q_estimate_his.append(q_estimate)
		q_true_his.append(q_true)

	now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
	# plot
	plt.figure(figsize=(10, 10))
	plt.title('state from -2 to 2 and action(-L*state)')
	plt.subplot(411)
	plt.plot(states)
	plt.title('state')

	plt.subplot(412)
	plt.plot(actions)
	plt.title('actions')
	# plt.show()

	plt.subplot(413)
	plt.plot(q_true_his)
	plt.title('true Q')
	# plt.show()

	plt.subplot(414)
	plt.plot(q_estimate_his)
	plt.title('estimate Q')
	plt.savefig(now+"q_true&estimate-state(-2,2)")
	plt.show()


	# q_estimate = 0.0
	# state = env.reset()
	# for i_steps in range(max_steps):
	# 	# print(state.getA()[0][0])
	# 	print("i_steps: {}".format(i_steps))
	# 	if np.isnan(state.getA()[0][0]):
	# 		exit()
	# 	action = agent.policy.get_action_with_L(state, L)

		
	# 	q_estimate = (agent.policy.q_state_action_func(state, action)).item()
	# 	q_estimate_his.append(q_estimate)
	# 	print("q_estimate: {}".format(q_estimate))
	# 	q_true = env.true_Qvalue(L, gamma, state, action)
	# 	q_true_his.append(q_true)
	# 	print("q_true: {}".format(q_true))

	# 	state_, reward, done, info = env.step(action)
	# 	# replay_buffer.store(state, action, reward, state_, done)
	# 	if i_steps%update_freq==0:
	# 		sample = replay_buffer.sample(batch_size)
	# 		error_list, new_weights = agent.train(sample)
	# 		true_weights = env.true_weights(L, gamma)
	# 		true_weights_his.append(true_weights)
	# 		true_estimate_error = np.linalg.norm(true_weights-new_weights)
	# 		print("true_estimate_error: {}".format(true_estimate_error))
	# 		true_estimate_error_history.append(true_estimate_error)
	# 	reward_history.append(reward)
	# 	state_history.append(state.getA()[0][0])
	# 	state = state_
	# 	print("")


	env.close()
	replay_buffer.reset()

if __name__ == '__main__':
	main()







