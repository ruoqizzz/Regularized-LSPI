# -*- coding: utf-8 -*-
'''
This file to check whether the q estimation is near to true q value 
for a specific policy
'''
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
import pickle
import os

# sample data files name for LQR
LQR_samples_filename = {
	2000: "samples/LQR/gaussian_actions_2000_2.pickle",
	5000: "samples/LQR/gaussian_actions_5000.pickle",
	10000: "samples/LQR/gaussian_actions_10000.pickle",
	20000: "samples/LQR/gaussian_actions_20000.pickle",
	"-22-100": "samples/LQR/states[-2,2]_100_L=0.1.pickle",
	"-22-1000": "samples/LQR/states[-2,2]_1000_L=0.1.pickle",
	"-22-10000": "samples/LQR/states[-2,2]_10000_L=0.1.pickle"
}


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="LQR", choices=["cliff-v0","CartPole-v0","inverted_pedulum","LQR","chain"])	# gym env to train
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=100, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_steps', default="2000", choices=["2000","5000","10000","20000"])
	parser.add_argument('--max_steps', default=500, type=int)
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2"])
	parser.add_argument('--reg_param', default=0.01, type=float)
	parser.add_argument('--rbf_sigma', default=0.001, type=float)
	# parser.add_argument('--batch_size', default=2000, type=int)
	parser.add_argument('--L', default=0.1, type=float)	# 0.0 means no random action
	

	args = parser.parse_args()
	params = vars(args)

	# env 
	env = LQREnv()
	params['n_actions'] = env.action_space.shape[0]
	params['state_dim'] = env.observation_space.shape[0]
	params['sample_max_steps'] = int(params['sample_max_steps'])
	# print(params['state_dim'])
	
	# basis function
	n_features = params['basis_function_dim']
	gamma = params['weight_discount']
	# params['basis_func'] = ExactBasis4LQR()
	params['basis_func'] = RBF_LQR(params['state_dim'], n_features, params['rbf_sigma'])

	# esitimate specific L
	L=np.matrix(params['L'])

	params['policy'] = ExactPolicy4LQR(params['basis_func'], L)

	# set the parameters for agent
	batch_size = params['sample_max_steps']
	max_steps = params['max_steps']

	agent = LSPIAgent(params)
	sample_filename = LQR_samples_filename[params['sample_max_steps']]
	# sample_filename = LQR_samples_filename["-22-10000"]
	f = open(sample_filename, 'rb')
	replay_buffer = pickle.load(f)

	sample = replay_buffer.sample(batch_size)
	print("length of sample: {}".format(len(sample)))
	error_list, new_weights = agent.train(sample)

	# logging
	true_weights_his = []
	true_estimate_error_history = []
	q_true_his = []
	q_estimate_his = []


	# for specific state
	# range of action
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
	# true_estimate_error = np.linalg.norm(true_weights-estimate_weights)
	# print("true_estimate_error: {}".format(true_estimate_error))

	now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

	# save data to file
	# note .item() only for one element
	dirname = "data/Estimation/state=" + str(state.item())+"/"
	try:
		os.mkdir(dirname)
	except OSError as error:  
		print(error)

	# # save q_true
	# filename = dirname + "q_true.pickle"
	# f = open(filename, 'wb')
	# pickle.dump(q_true_his, f)
	# f.close()
	# save q_estimate

	if params['basis_func'].name()[:3] == 'RBF':
		filename = dirname + params['basis_func'].name()+"-"+str(params['basis_function_dim'])+"-"+params['reg_opt']+"-"+str(params['reg_param'])+".pickle"
	else:
		filename = dirname + params['basis_func'].name()+".pickle"
	f1 = open(filename, 'wb')
	pickle.dump(q_estimate_his, f1)
	f1.close()


	plt.figure(figsize=(8, 6))
	plt.subplot(211)
	plt.plot(actions, q_estimate_his)
	plt.title('q estimate')

	plt.subplot(212)
	plt.plot(actions, q_true_his)
	plt.title('q true')
	# plt.savefig("images/rbf-lqr/"+str(n_features)+"-"+now+"q_true&estimate-action(-1,1)")
	plt.show()


	# for state range
	state_low = -10.0
	state_high = 10.0
	states = np.linspace(state_low, state_high, 100)
	actions = []
	true_weights_his = []
	true_estimate_error_history = []
	q_true_his = []
	q_estimate_his = []

	for i in range(len(states)):
		state = np.matrix(states[i])
		action = -L*state
		actions.append(action.item())
		q_estimate = (agent.policy.q_state_action_func(state, action)).item()
		q_true = env.true_Qvalue(L, gamma, state, action)
		q_state = env.true_Qvalue_state(L, gamma, state)

		q_estimate_his.append(q_estimate)
		q_true_his.append(q_true)


	now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

	# save estimate data to file
	dirname = "data/Estimation/states[" + str(state_low)+","+str(state_high)+"]/"
	try:  
		os.mkdir(dirname)
	except OSError as error:
		print(error)  
	# # q_true
	# filename = dirname + "q_true.pickle"
	# f = open(filename, 'wb')
	# pickle.dump(q_true_his, f)
	# f.close()

	# estimate
	if params['basis_func'].name()[:3]=='RBF':
		filename = dirname + params['basis_func'].name()+"-"+str(params['basis_function_dim'])+"-"+params['reg_opt']+"-"+str(params['reg_param'])+".pickle"
	else:
		filename = dirname + params['basis_func'].name()+".pickle"
	f1 = open(filename, 'wb')
	pickle.dump(q_estimate_his, f1)
	f1.close()

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
	plt.plot(states, q_true_his)
	plt.title('true Q')
	# plt.show()

	plt.subplot(414)
	plt.plot(states, q_estimate_his)
	plt.title('estimate Q')
	# plt.savefig(now+"q_true&estimate-state(-2,2)")
	# plt.show()


	# plt.savefig("images/rbf-lqr/L2/20-"+str(n_features)+"-"+now+"q_true&estimate-state(-2,2)")
	plt.show()

	env.close()
	replay_buffer.reset()

if __name__ == '__main__':
	main()







