import argparse
from lspi import LSPIAgent

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="CartPole-v0", choices=["CartPole-v0", "MoutainCar-v0"])	# gym env to train


	args = parser.parse_args()
	params = vars(args)

	env = gym.make(params['env_name'])


	# get the parameters of env
	n_actions = env.action_space.n
	n_observations = env.observation_space.shape[0]
	params['n_actions'] = n_actions
	params['n_observations'] = n_observations


	n_episode = params['episode_num']

	agent = LSPIAgent()
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