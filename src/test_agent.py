import copy
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import time

def test_agent4LQR(agent, env, gamma, ifshow=False):
	# states = np.linspace(-10,10,500)
	states = np.linspace([-4]*env.m, [4]*env.m, 801)
	trueL = env.optimal_policy_L(gamma)
	
	actions_true = []
	for i in range(len(states)):
		state = np.matrix(states[i].reshape(env.m,1))
		# action = agent.policy.get_best_action(state)
		# actions_estimate.append(action)
		actions_true.append((-trueL*state).item())
	# print(actions_true)
	actions_estimate = agent.policy.get_best_action(states)

	# plot policy	
	now = time.strftime("%Y-%m-%d",time.localtime(time.time()))
	now2 = time.strftime("%H_%M_%S",time.localtime(time.time()))
	path = "data/LQR/"+now+"/"+str(agent.opt)+"-"+str(agent.reg_param)+"-BF"+str(agent.policy.basis_func.size())+'-'+agent.policy.basis_func.name()+"/"+now2+"/"
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)   
	fn = path+"policy.pickle"
	f = open(fn, 'wb')
	pickle.dump(actions_estimate, f)
	f.close()

	# plot
	plt.plot(states, actions_estimate)
	# print(actions_true)
	plt.xlabel('states')
	plt.ylabel('actions')
	plt.ylim(-3,3)
	plt.plot(states, actions_true)
	plt.legend(('estimate', 'true'))
	# plt.show()
	pltfn = path+"policy.png"
	# f = open(fn, 'wb')
	plt.savefig(pltfn,dpi=300)
	if ifshow:
		plt.show()
	plt.clf()

	# test on env
	state = env.reset()
	state_o = copy.copy(state)

	env_o = copy.deepcopy(env)

	agent_state_his = []
	optimal_state_his = []

	agent_action_his = []
	optimal_action_his = []

	n_episode = 100
	for i in range(n_episode):
		agent_state_his.append(state.item())
		optimal_state_his.append(state_o.item())

		# print(state)
		# print(state_o)

		action = agent.policy.get_best_action(np.array([state]))[0]
		action_o = -trueL*state_o
		# print("action ", action)
		# print("action_o ", action_o)

		agent_action_his.append(action)
		optimal_action_his.append(action_o.item())

		state_, reward, done, info = env.step(action)
		state_o_, reward_o, done_o, info_o = env_o.step(action_o)

		state = state_
		state_o = state_o_


	# plot
	plt.plot(agent_state_his, label='estimate')
	plt.plot(np.zeros(n_episode), 'k--')
	plt.plot(optimal_state_his, label='optiomal')
	# print(actions_true)
	plt.ylabel('state')
	plt.xlabel('episode')
	plt.legend(loc='upper right')
	# plt.show()
	pltfn = path+"state.png"
	# f = open(fn, 'wb')
	plt.savefig(pltfn,dpi=300)
	if ifshow:
		plt.show()

	plt.clf()

	fn = path+"state_his.pickle"
	f = open(fn, 'wb')
	pickle.dump(agent_state_his, f)
	f.close()

	plt.plot(agent_action_his, label='estimate')
	plt.plot(optimal_action_his, label='optiomal')
	# print(actions_true)
	plt.ylabel('action')
	plt.xlabel('episode')
	plt.legend(loc='upper right')
	# plt.show()
	pltfn = path+"action.png"
	# f = open(fn, 'wb')
	plt.savefig(pltfn,dpi=300)
	if ifshow:
		plt.show()
	fn = path+"action_his.pickle"
	f = open(fn, 'wb')
	pickle.dump(agent_state_his, f)
	f.close()


