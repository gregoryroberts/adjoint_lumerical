from context import inverse_design
from inverse_design import CMOSBayerFilter
import numpy as np
import matplotlib.pyplot as plt

np_seed = 624234
np.random.seed(np_seed)

design = np.random.random((24, 24, 24)) - 0.5

bayer_filter = CMOSBayerFilter.CMOSBayerFilter(design.shape, [1., 1.5], 0.5)

for epoch in range(0, 4):
	for iter in range(0, 10):
		bayer_filter.update_filters(epoch)
		rand_grad = np.random.random((24, 24, 24)) - 0.5
		rand_grad = 0.5 * (rand_grad + rand_grad.swapaxes(0, 1))
		bayer_filter.step(rand_grad, 0.5)

# bayer_filter.update_filters(5)
# bayer_filter.update_permittivity()

sim_device = bayer_filter.get_permittivity()
# print(bayer_filter.w[4])

for z in range(0, 24):
	plt.subplot(6, 4, z + 1)
	plt.imshow(sim_device[:, :, z] - sim_device[:, :, z].swapaxes(0, 1))
	print(z)
	print(sim_device[:, :, z])
	plt.colorbar()
plt.show()


