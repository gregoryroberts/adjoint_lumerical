import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../inverse_design')))

import CMOSBayerFilter

import numpy as np
import time


def f(permittivity):
	return ( np.sum(permittivity**2) + (np.sum(permittivity**2))**2 )

def grad_f(permittivity):
	return ( 2 * permittivity * (1 + 2 * np.sum(permittivity**2) ) )

np.random.seed(423423)

dim = 20
bayer_filter = CMOSBayerFilter.CMOSBayerFilter(np.array([dim, dim, dim]), [1.0, 1.5], 0.25, 4)

bayer_filter.set_design_variable(np.random.random((dim, dim, dim)))

device = bayer_filter.get_permittivity()
device1 = device.copy()

h = 1e-5

start = f(device)

loc = 8

design = bayer_filter.get_design_variable()
design[loc, loc, loc] += h
bayer_filter.set_design_variable(design)
device = bayer_filter.get_permittivity()

end = f(device)

design[loc, loc, loc] -= h
bayer_filter.set_design_variable(design)
device = bayer_filter.get_permittivity()


grad_device = grad_f(device)
grad_design = bayer_filter.backpropagate(grad_device)

fd_grad = (end - start) / h

print('fd grad = ' + str(fd_grad))
print('grad design = ' + str(grad_design[loc, loc, loc]))


