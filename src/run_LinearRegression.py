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
import pickle
from sklearn.linear_model import LinearRegression
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
	parser.add_argument('--basis_function_dim', default=30, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_steps', default="2000", choices=["2000","5000","10000","20000"])
	parser.add_argument('--max_steps', default=500, type=int)
	# parser.add_argument('--batch_size', default=2000, type=int)
	parser.add_argument('--L', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2"])
	parser.add_argument('--reg_param', default=0.01, type=float)
	args = parser.parse_args()
	params = vars(args)

	env = LQREnv()
	params['n_actions'] = env.action_space.shape[0]
	params['state_dim'] = env.observation_space.shape[0]
	params['sample_max_steps'] = int(params['sample_max_steps'])
	# print(params['state_dim'])
	# params['basis_func'] = ExactBasis4LQR()
	# basis_function = ExactBasis4LQR()
	n_features = params['basis_function_dim']
	gamma = params['weight_discount']
	basis_function = RBF_LQR(params['state_dim'], n_features, 0.001)
	params['basis_func'] = basis_function

	# esitimate specific L
	L=np.matrix(params['L'])


	# set the parameters for agent
	# use all the samples in buffer
	batch_size = params['sample_max_steps']
	max_steps = params['max_steps']

	sample_filename = LQR_samples_filename[params['sample_max_steps']]
	# sample_filename = LQR_samples_filename[-22]
	f = open(sample_filename, 'rb')
	replay_buffer = pickle.load(f)
	sample = replay_buffer.sample(batch_size)

	# samples to true Q value
	phi_list = []
	qTrue_list = []

	for i in range(len(sample)):
		print("i: {}".format(i))
		s = sample[i]
	# for s in sample:
		state = s.state
		action = s.action
		reward = s.reward
		next_state = s.next_state
		done = s.done
		# print("state: {}".format(state))
		# print("action: {}".format(action))
		phi = basis_function.evaluate(state, action)
		# print("phi: {}".format(phi))
		phi_list.append(phi)
		qTrue = env.true_Qvalue(L, gamma, state, action)
		# print("qTrue: {}".format(qTrue))
		qTrue_list.append(qTrue)

	phi_list = np.array(phi_list)
	# print("phi_list shape: {}".format(phi_list.shape))
	print("phi_list: {}".format(phi_list[:10]))
	qTrue_list = np.array(qTrue_list)
	print("qTrue_list: {}".format(qTrue_list[:10]))
	# print("qTrue_list shape: {}".format(qTrue_list.shape))
	reg = LinearRegression().fit(phi_list, qTrue_list)
	# print("reg.get_params(): {}".format(reg.get_params()))

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
		q_estimate = reg.predict([basis_function.evaluate(state, action)])
		q_true = env.true_Qvalue(L, gamma, state, action)
		q_state = env.true_Qvalue_state(L, gamma, state)

		q_estimate_his.append(q_estimate)
		q_true_his.append(q_true)

	now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

	# save estimate data to file
	dirname = "data/Regression/states[" + str(state_low)+","+str(state_high)+"]/"
	try:  
		os.mkdir(dirname)
	except OSError as error:
		print(error)  
	# # q_true
	filename = dirname + "q_true.pickle"
	f = open(filename, 'wb')
	pickle.dump(q_true_his, f)
	f.close()

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
	plt.show()

	env.close()
	replay_buffer.reset()

if __name__ == '__main__':
	main()





