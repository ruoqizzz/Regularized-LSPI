import argparse
from lspi import LSPIAgent
from replay_buffer import ReplayBuffer
import gym
import gymgrid

EARLY_SAMPLES = 500

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default="cliff-v0", choices=["cliff-v0","CartPole-v0", "MoutainCar-v0"])	# gym env to train
	parser.add_argument('--episode_num', default=5000, type=int)
	parser.add_argument('--weight_discount', default=0.99, type=float)	# note: 1.0 only for finite
	parser.add_argument('--exploration', default=0.0, type=float)	# 0.0 means no random action
	parser.add_argument('--basis_function_dim', default=5, type=int)
	parser.add_argument('--stop_criterion', default=10**-5, type=float)
	parser.add_argument('--learning_start', default=500, type=int)
	parser.add_argument('--batch_size', default=500, type=int)
	args = parser.parse_args()
	params = vars(args)

	env = gym.make(params['env_name'])


	# set the parameters for agent
	state_dim = env.observation_space.shape[0]
	# action_dim = 1
	params['n_actions'] = env.action_space.n
	params['state_dim'] = state_dim
	batch_size = params['batch_size']
	n_episode = params['episode_num']
	agent = LSPIAgent(params)
	total_steps = 0
	replay_buffer = ReplayBuffer()
	history = []
	for i_episode in range(n_episode):
		state = env.reset()
		done  = False
		total_reward = 0
		i_episode_steps = 0
		while True:
			i_episode_steps += 1
			total_steps += 1
			# if total_steps > params['learning_start']:
			# 	env.render()
			action = agent.get_action(state)
			state_, reward, done, info = env.step(action)
			replay_buffer.store(state, action, reward, state_, done)
			total_reward += reward
			if total_steps > params['learning_start'] and total_steps%batch_size==0:
				sample = replay_buffer.sample(batch_size)
				error_list = agent.train(sample)
			if done:
				print(total_reward)
				history.append(total_reward)
				break
	env.close()
	replay_buffer.reset()



if __name__ == '__main__':
	main()