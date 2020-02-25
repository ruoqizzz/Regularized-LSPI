# -*- coding: utf-8 -*-
'''
	Do some simple test here
'''
from env.inverted_pendulum import InvertedPendulumEnv
from env.chain import ChainEnv
from env.linear_quadratic_regulator import LQREnv
import gym
import time
import numpy as np 

def test_inverted_pendulum():
	env = InvertedPendulumEnv()
	# env = gym.make("Pendulum-v0")
	env.reset()
	for _ in range(2000):
		env.render()
		time.sleep(0.5)
		# action = env.action_space.sample()
		action = 2
		obs, reward, done, info = env.step(action) # take a random action
		# if done:
		# 	print("Failed")
		#	break
	env.close()

def test_chain():
	env = ChainEnv()
	# env = gym.make("Pendulum-v0")
	obs_first = env.reset()
	print("obs_first: {}".format(obs_first))
	for _ in range(100):
		# env.render()
		time.sleep(0.5)
		# action = env.action_space.sample()
		action = 0
		obs, reward, done, info = env.step(action) # take a random action
		print("obs: {}".format(obs))
		if done:
			print("Done")
			break
	env.close()

def test_AwithOptimalL():
	A_values= list(np.arange(0.1,1.0, 0.1))
	L = []
	for a in A_values:
		env = LQREnv(A=np.matrix(a))
		L.append(env.optimal_policy_L(0.99).item())
	import matplotlib.pyplot as plt
	plt.plot(A_values, L)
	plt.xlabel('value A')
	plt.ylabel('optimal L')
	plt.show()

if __name__ == '__main__':
	# test_inverted_pendulum()
	# test_chain()
	test_AwithOptimalL()
