# -*- coding: utf-8 -*-
'''
This file to check the resulf of RBF filter using true Q value:
	1. for estimation, if it fits well?
	2. for improvement, if it get the optimal control L?
Details:
	In order use the true Q, need to rewrite the lspi agent and class lstdq where the weights is updated
	Also, everytime update the A and b in lstq. Just use the true weights from environment
'''


