import argparse
from lspi import LSPIAgent

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="CartPole-v0", choices=["CartPole-v0", "MoutainCar-v0"])	# gym env to train
	parser.add_argument('--episode_num', default=300, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.0, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=5, type=int)


	args = parser.parse_args()
	params = vars(args)

	env = gym.make(params['env_name'])


	# set the parameters for agent
	state_dim = env.observation_space.shape[0]
	# action_dim = 1
	params['n_actions'] = env.action_space.n
	params['state_dim'] = state_dim

	n_episode = params['episode_num']
	agent = LSPIAgent(params)
	total_steps = 0

	for i_episode in range(n_episode):
		observation = env.reset()
		done = False
		total_reward = 0
		i_episode_steps = 0
		while True:
			i_episode_steps +=1
			total_steps += 1
			env.render()
			action = agent.get_action(observation, epsilon)
			observation_, reward, done, info = env.step(action)

			pass

			if done:
				print(total_reward)
				history.append(total_reward)
				break
	env.close()



if __name__ == '__main__':
	main()