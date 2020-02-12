'''
	Do some simple test here
'''
from inverted_pendulum import InvertedPendulumEnv
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
		# 	time.sleep(1.0)
	env.close()


if __name__ == '__main__':
	test_inverted_pendulum()