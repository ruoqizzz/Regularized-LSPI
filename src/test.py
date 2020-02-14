# -*- coding: utf-8 -*-
'''
	Do some simple test here
'''
from env.inverted_pendulum import InvertedPendulumEnv
from env.chain import ChainEnv
import gym
import time

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


if __name__ == '__main__':
	test_inverted_pendulum()
	test_chain()
