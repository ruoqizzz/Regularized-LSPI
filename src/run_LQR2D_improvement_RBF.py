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

# sample data files name for LQR2D

LQR2D_samples_filename = {
	2000: "samples/LQR2D/gaussian_actions_2000.pickle",
	5000: "samples/LQR2D/gaussian_actions_5000.pickle",
}
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=100, type=int)
	parser.add_argument('--stop_criterion', default=10**-3, type=float)
	parser.add_argument('--sample_max_steps', default="10000", choices=["2000","5000","10000","20000"])
	parser.add_argument('--max_steps', default=20, type=int)
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2", "wl1", "none"])
	parser.add_argument('--reg_param', default=0.001, type=float)
	parser.add_argument('--rbf_sigma', default=0.01, type=float)
	# parser.add_argument('--batch_size', default=2000, type=int)
	parser.add_argument('--L', default=0.1, type=float)	# 0.0 means no random action
	

	args = parser.parse_args()
	params = vars(args)

	# env 
	# env = LQREnv()
	A = np.matrix([[0.9,0.],[0.08,0.9]])
	B = np.matrix([[0.1],[0.6]])
	Z1 = np.matrix([[1,0],[0,0]])
	Z2 = 0.1
	noise_cov = np.matrix([[1,0],[0,1]])
	env = LQREnv(A=A,B=B,Z1=Z1,Z2=Z2,noise_cov=noise_cov)
	params['n_actions'] = env.action_space.shape[0]
	params['state_dim'] = env.observation_space.shape[0]
	params['sample_max_steps'] = int(params['sample_max_steps'])
	# print(params['state_dim'])
	
	# basis function
	n_features = params['basis_function_dim']
	gamma = params['weight_discount']
	# params['basis_func'] = ExactBasis4LQR()
	params['basis_func'] = RBF_LQR([params['state_dim'], params['n_actions']], n_features, params['rbf_sigma'])

	params['policy'] = RBFPolicy4LQR(params['basis_func'])

	# set the parameters for agent
	batch_size = params['sample_max_steps']
	max_steps = params['max_steps']

	agent = LSPIAgent(params)
	sample_filename = LQR2D_samples_filename[params['sample_max_steps']]
	# sample_filename = LQR_samples_filename["-22-10000"]
	f = open(sample_filename, 'rb')
	replay_buffer = pickle.load(f)

	sample = replay_buffer.sample(batch_size)
	print("length of sample: {}".format(len(sample[0])))
	error_list, new_weights = agent.train(sample)

	# states = np.linspace(-10,10,500)
	states = np.linspace([-10]*env.m, [10]*env.m, 500)
	trueL = env.optimal_policy_L(gamma)
	
	actions_true = []
	for i in range(len(states)):
	    state = np.matrix(states[i].reshape(env.m,1))
	    # action = agent.policy.get_best_action(state)
	    # actions_estimate.append(action)
	    actions_true.append((-trueL*state).item())
	# print(actions_true)
	actions_estimate = agent.policy.get_best_action(states)
	# save agent
	
	now = time.strftime("%Y-%m-%d",time.localtime(time.time()))
	now2 = time.strftime("%H_%M_%S",time.localtime(time.time()))
	path = "data/LQR2D/"+now
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)   
	fn = path+"/data-"+str(params['reg_opt'])+"-"+str(params['reg_param'])+"-BF"+str(n_features)+"-"+params['basis_func'].name()+"-"+now2+".pkl"
	f = open(fn, 'wb')
	pickle.dump(actions_estimate, f)
	f.close()

	# plot
	plt.plot(states, actions_estimate, label='estimate')
	# print(actions_true)
	plt.plot(states, actions_true, label='true')
	plt.legend(loc='upper right')
	pltfn = path+"/data-"+str(params['reg_opt'])+"-"+str(params['reg_param'])+"-BF"+str(n_features)+"-"+params['basis_func'].name()+"-"+now2+".png"
	# f = open(fn, 'wb')
	plt.savefig(pltfn, dpi=300)
	# plt.show()
	# clean
	env.close()


if __name__ == '__main__':
	main()








