# policy greedy
import numpy as np
class GreedyPolicy(object):
	"""docstring for GreedyPolicy"""
	def __init__(self, basis_func, n_actions, epsilon):
		super(GreedyPolicy, self).__init__()
		self.basis_func = basis_func
		self.epsilon = epsilon
		self.n_basis_func = self.basis_func.size()
		self.n_actions = n_actions
		self.actions = list(range(n_actions))
		self.weights = np.random.uniform(-1.0, 1.0, size=(self.n_basis_func, 1))


	def q_state_action_func(self, state, action):
        # Q(s, a; w) = sum (pi(s, a) * weights)
        # # basis functions pi(s, a)
		vector_basis = self.basis_func.evaluate(state,action)
		return np.dot(vector_basis, self.weights)  # pi(s, a) * weights

	def get_best_action_epsilon(self, state, opt='random'):
		q_state_actions = [self.q_state_action_func(state, a) for a in self.actions]
		q_state_actions = np.reshape(q_state_actions, [len(q_state_actions), 1]) # convert to column vector
		index = np.argmax(q_state_actions)
		q_max = q_state_actions[index]
		best_action = self.actions[index]
		rng = np.random.default_rng()
		if rng.random() < self.epsilon:
			return self.actions[index]
		else:
			return random.sample(self.actions,1)
	
	def update_weights(self, new_weights):
		self.weights = new_weights
	