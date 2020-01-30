# agent
from basis_function import RBF
from policy import GreedyPolicy
class LSPIAgent(object):
	"""docstring for LSPIAgent"""
	def __init__(self, param):
		super(LSPIAgent, self).__init__()
		self.state_dim = param['state_dim']
		self.n_actions = param['n_actions']
		self.basisfunc_dim = param['basisfunc_dim']
		self.gamma = param['weight_discount']
		
		self.basis_function = RBF(self.state_dim, self.basisfunc_dim, self.n_actions, self.gamma)
		self.num_bf = self.basis_function.size()

		epsilon = 1-param['exploration']
		self.policy = GreedyPolicy(epsilon)
		self.lstdq = LSTDQ(self.basis_function, gamma)
		self.stop_criterion = 10**-5


	def train(self, sample):
		error = float('inf')
		error_his = []
		while error > self.stop_criterion:
			new_weights = self.lstdq.update_weights(sample, self.policy)
			error = np.linalg.norm((new_weights - self.policy.weights))
			error_his.append(error)
			self.policy.update_weights(new_weights)
		return error_his


	def get_action():
		return self.policy.get_best_action()


class LSTDQ(object):
	"""docstring for LSTDQ"""
	def __init__(self, arg):
		super(LSTDQ, self).__init__()
		self.arg = arg
		
	def update_weights():
		pass

