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
	2000: "samples/LQR/gaussian_actions_2000.pickle",
	5000: "samples/LQR/gaussian_actions_5000.pickle",
}

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="LQR", choices=["cliff-v0","CartPole-v0","inverted_pedulum","LQR","chain"])	# gym env to train
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=50, type=int)
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
	
	n_features = params['basis_function_dim']
	gamma = params['weight_discount']
	# params['basis_func'] = ExactBasis4LQR()
	# basis_function = ExactBasis4LQR()

	basis_function = RBF_LQR([params['state_dim'],params['n_actions']], n_features, 0.001)
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
	samples = replay_buffer.sample(batch_size)

	# samples to true Q value
	phi_list = []
	qTrue_list = []

	states = samples[0]
	actions = samples[1]
	rewards = samples[2]
	next_states = samples[3]
	dones = samples[4]

	phi_list = basis_function.evaluate(states,actions)

	for i in range(len(states)):
		print("i: {}".format(i))
		s = states[i]
		# print("state: {}".format(state))
		# print("action: {}".format(action))
		qTrue = env.true_Qvalue(L, gamma, states[i], actions[i])
		# print("qTrue: {}".format(qTrue))
		qTrue_list.append(qTrue)

	phi_list = np.array(phi_list)
	# print("phi_list shape: {}".format(phi_list.shape))
	# print("phi_list: {}".format(phi_list[:10]))
	qTrue_list = np.array(qTrue_list)
	# print("qTrue_list: {}".format(qTrue_list[:10]))
	# print("qTrue_list shape: {}".format(qTrue_list.shape))
	reg = LinearRegression().fit(phi_list, qTrue_list)
	# print("reg.get_params(): {}".format(reg.get_params()))

	# for state range
	state_low = -10.0
	state_high = 10.0
	states = np.linspace(state_low, state_high, 100)
	actions = []
	# true_weights_his = []
	true_estimate_error_history = []
	q_true_his = []
	q_estimate_his = reg.predict(basis_function.evaluate(states, -L.item()*states))

	for i in range(len(states)):
		state = np.matrix(states[i])
		action = -L*state
		actions.append(action.item())
		q_true = env.true_Qvalue(L, gamma, state, action)
		# q_state = env.true_Qvalue_state(L, gamma, state)
		q_true_his.append(q_true)

	now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

	# save estimate data to file
	dirname = "data/Regression/states[" + str(state_low)+","+str(state_high)+"]/"
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
	# if params['basis_func'].name()[:3]=='RBF':
	# 	filename = dirname + params['basis_func'].name()+"-"+str(params['basis_function_dim'])+"-"+params['reg_opt']+"-"+str(params['reg_param'])+".pickle"
	# else:
	# 	filename = dirname + params['basis_func'].name()+".pickle"
	# f1 = open(filename, 'wb')
	# pickle.dump(q_estimate_his, f1)
	# f1.close()

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

	# for specific state
	# range of action
	state = np.matrix(-1.)
	actions = np.linspace(-6,6, 100)
	q_true_his = []
	
	q_estimate_his = reg.predict(basis_function.evaluate(np.full(len(actions), state), actions))
	for i in range(len(actions)):
		action = np.matrix(actions[i])
		# print("q_estimate: {}".format(q_estimate))
		q_true = env.true_Qvalue(L, gamma, state, action)
		# print("q_true: {}".format(q_true))
		q_true_his.append(q_true)

	# true_weights_scala = env.true_weights_scala(L, gamma)
	# print("true_weights_scala: {}".format(true_weights_scala))
	# estimate_weights = agent.policy.weights
	# print("estimate_weights: {}".format(estimate_weights))
	# true_estimate_error = np.linalg.norm(true_weights_scala-estimate_weights)
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

	# if params['basis_func'].name()[:3] == 'RBF':
	# 	filename = dirname + params['basis_func'].name()+"-"+str(params['basis_function_dim'])+"-"+params['reg_opt']+"-"+str(params['reg_param'])+".pickle"
	# else:
	# 	filename = dirname + params['basis_func'].name()+".pickle"
	# f1 = open(filename, 'wb')
	# pickle.dump(q_estimate_his, f1)
	# f1.close()

	print("q_estimate_his: {}".format(q_estimate_his))
	plt.figure(figsize=(8, 6))
	plt.subplot(211)
	plt.plot(actions, q_estimate_his)
	plt.title('q estimate')

	plt.subplot(212)
	plt.plot(actions, q_true_his)
	plt.title('q true')
	# plt.savefig("images/rbf-lqr/"+str(n_features)+"-"+now+"q_true&estimate-action(-1,1)")
	plt.show()

	env.close()
	replay_buffer.reset()

if __name__ == '__main__':
	main()




