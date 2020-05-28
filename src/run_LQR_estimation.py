# -*- coding: utf-8 -*-
'''
This file to check whether the q estimation is near to true q value 
for a specific policy
'''
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
import pickle
import os

# sample data files name for LQR
LQR_samples_filename = {
	2000: "samples/LQR/gaussian_actions_2000.pickle",
	5000: "samples/LQR/gaussian_actions_5000.pickle",
}


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=40, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--sample_max_steps', default="5000", choices=["2000","5000"])
	parser.add_argument('--max_steps', default=500, type=int)
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2", "wl1", "none"])
	parser.add_argument('--reg_param', default=0.001, type=float)
	parser.add_argument('--rbf_sigma', default=0.01, type=float)
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
	params['basis_func'] = RBF_LQR([params['state_dim'], params['n_actions']], n_features, params['rbf_sigma'])
	
	# esitimate specific L
	L=np.matrix(params['L'])

	# params['policy'] = ExactPolicy4LQR(params['basis_func'], L)
	params['policy'] = RBFPolicy4LQR(params['basis_func'], L)
	# set the parameters for agent
	batch_size = params['sample_max_steps']
	max_steps = params['max_steps']

	agent = LSPIAgent(params)
	sample_filename = LQR_samples_filename[params['sample_max_steps']]
	f = open(sample_filename, 'rb')
	replay_buffer = pickle.load(f)

	samples = replay_buffer.sample(batch_size)
	print("length of sample: {}".format(len(samples[0])))
	error_list, new_weights = agent.train(samples)



	# for specific state
	# range of action
	for si in range(-10,10,5):
		si = -1.0
		true_estimate_error_history = []
		q_true_his = []
		q_estimate_his = []

		state = np.matrix(si)
		actions = np.linspace(-6,6, 100)

		q_estimate_his = agent.policy.q_state_action_func(np.full(len(actions), state), actions)
		for i in range(len(actions)):
			action = np.matrix(actions[i])
			# q_estimate = agent.policy.q_state_action_func(state, action)[0]
			# q_estimate_his.append(q_estimate)
			# print("q_estimate: {}".format(q_estimate))
			q_true = env.true_Qvalue(L, gamma, state, action)
			# print("q_true: {}".format(q_true))
			q_true_his.append(q_true)

		true_weights_scala = env.true_weights_scala(L, gamma)
		print("true_weights_scala: {}".format(true_weights_scala))
		estimate_weights = agent.policy.weights
		print("estimate_weights: {}".format(estimate_weights))
		true_estimate_error = np.linalg.norm(true_weights_scala-estimate_weights)
		print("true_estimate_error: {}".format(true_estimate_error))

		# now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

		# save data to file
		# note .item() only for one element
		# dirname = "data/Estimation/state=" + str(state.item())+"/"
		# try:
		# 	os.mkdir(dirname)
		# except OSError as error:  
		# 	print(error)

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

		qe_index = np.argmax(q_estimate_his)
		qt_index = np.argmax(q_true_his)

		plt.figure(figsize=(10, 8))
		plt.subplot(211)
		ax = plt.gca()
		plt.plot(actions, q_estimate_his)
		plt.scatter(actions[qe_index], q_estimate_his[qe_index], c='r')
		plt.xlabel('actions')
		plt.ylabel('q value')
		plt.title('estimate q value')
		ax.xaxis.set_label_coords(1.02, -0.035)
		plt.subplot(212)
		plt.plot(actions, q_true_his)
		plt.scatter(actions[qt_index], q_true_his[qt_index], c='r')

		plt.title('true q value')
		# plt.savefig("images/rbf-lqr/"+str(n_features)+"-"+now+"q_true&estimate-action(-1,1)")
		plt.show()
		

	env.close()
	replay_buffer.reset()

if __name__ == '__main__':
	main()







