

class LQREnv(object):
	"""docstring for LQREnv"""
	def __init__(self, arg):
		super(LQREnv, self).__init__()
		self.arg = arg

	def seed(self, seed=None):
        
        return [seed]

    def step(self, action):


    	return self._get_obs(), reward, done, {}

    def reset(self):
    	pass

    def _get_obs(self):
    	pass

    def close(self):
    	pass
		