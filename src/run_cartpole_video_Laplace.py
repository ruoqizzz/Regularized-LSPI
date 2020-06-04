# -*- coding: utf-8 -*-
import argparse
from lspi import *
from replay_buffer import ReplayBuffer
import gym
from env.inverted_pendulum import InvertedPendulumEnv
from env.linear_quadratic_regulator import LQREnv
from env.chain import ChainEnv
from basis_func import *
import time
from collect_samples import *
from policy import *
import matplotlib.pyplot as plt
import pickle
from gym import wrappers
import os

def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--env_name', default="CartPole-v0", choices=["cliff-v0","CartPole-v0","inverted_pedulum","chain"])	# gym env to train
	parser.add_argument('--episode_num', default=1000, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.1, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=200, type=int)
	parser.add_argument('--stop_criterion', default=10**-3, type=float)
	# parser.add_argument('--batch_size', default=1000, type=int)
	# parser.add_argument('--update_freq', default=1000, type=int)
	parser.add_argument('--samples_episodes', default="200", choices=["200","400","600","1000"])
	parser.add_argument('--reg_opt', default="l2", choices=["l1","l2","wl1","none"])
	parser.add_argument('--reg_param', default=0.01, type=float)
	parser.add_argument('--rbf_sigma', default=0.5, type=float)
	parser.add_argument('--lspi_iteration', default=100, type=int)
	
	args = parser.parse_args()
	params = vars(args)

	env = gym.make('CartPole-v0')

	# set the parameters for agent
	# batch_size = params['batch_size']
	# update_freq = params['update_freq']
	n_episode = params['episode_num']
	gamma = params['weight_discount']
	n_features = params['basis_function_dim']
	lspi_iteration = params['lspi_iteration']

	params['n_actions'] = env.action_space.n
	params['state_dim'] = env.observation_space.shape[0]
	

	samples_episodes = params['samples_episodes']
	# fn = "samples/CartPole/CartPole"+samples_episodes+"-2.pickle"
	# f = open(fn, 'rb')
	# replay_buffer = pickle.load(f)
	# f.close()
	replay_buffer = collect_samples_maxepisode(env, int(samples_episodes))
	i_samples = replay_buffer.sample(replay_buffer.num_buffer)

	L_vec = 1.1*np.max(np.abs(i_samples[0][:,2:4]), axis=0).flatten()
	basis_func= Laplace(n_features, L_vec, params['n_actions'])
	# print(basis_func.size())
	params['basis_func'] = basis_func

	policy = GreedyPolicy(params['basis_func'], params['n_actions'], 1-params['exploration'])
	params['policy'] = policy
	agent = BellmanAgent(params, n_iter_max=15)

	
	agent.train(i_samples)
	
	now = time.strftime("%Y-%m-%d",time.localtime(time.time()))
	now2 = time.strftime("%H_%M_%S",time.localtime(time.time()))
	# path = "data/CartPole/"+now+"/"+str(agent.opt)+"-"+str(agent.reg_param)+"-LP"+str(agent.policy.basis_func.size())+'-'+agent.policy.basis_func.name()+'-eps'+str(samples_episodes)+"/"+now2+"/"
	path = "data/CartPole/[theta, theta_dot]/"+str(agent.opt)+"-"+str(agent.reg_param)+"-LP"+str(agent.policy.basis_func.size())+'-'+agent.policy.basis_func.name()+'-eps'+str(samples_episodes)+"/"+now2+"/"
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
	print(path)
	# print(path)
	env = wrappers.Monitor(env, path, video_callable=False)

	state = env.reset()
	done  = False
	total_reward = 0
	i_episode_steps = 0
	history = []
	x_history = []
	xdot_history = []
	theta_history = []
	thetadot_history = []
	while True:
		# env.render()
		# dim of state!
		x_history.append(state[0])
		xdot_history.append(state[1])
		theta_history.append(state[2])
		thetadot_history.append(state[3])

		state = np.reshape(state[2:4], (1,2))
		# state = np.reshape(state,(1,4))
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
			# print("x: {}".format(state_[0]))
			# history.append(total_reward)
			# print("i_episode_steps {}".format(i_episode_steps))
			print("total_reward {}".format(total_reward))
			# time.sleep(0.1)
			break
	pickle.dump(x_history, open(path+'x_history.pickle', 'wb'))
	pickle.dump(xdot_history, open(path+'xdot_history.pickle', 'wb'))
	pickle.dump(theta_history, open(path+'theta_history.pickle', 'wb'))
	pickle.dump(thetadot_history, open(path+'thetadot_history.pickle', 'wb'))

	f = open(path+'total_reward.txt', 'w')
	f.write(str(total_reward))
	f.close()
	
	plt.plot(x_history)
	# print(actions_true)
	plt.ylabel('x')
	plt.xlabel('episodes')
	# plt.plot(np.ones(len(x_history))*2.4, 'k--')
	plt.savefig(path+'x_history.png',dpi=300)
	plt.clf()

	plt.plot(theta_history)
	plt.ylabel('theta')
	plt.xlabel('episodes')
	plt.savefig(path+'thteta_history.png',dpi=300)
	plt.clf()



if __name__ == '__main__':
	main()
