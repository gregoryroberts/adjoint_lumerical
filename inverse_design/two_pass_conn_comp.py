import numpy as np
# import scipy.io as sio
import skimage.morphology as skim
# import time

def six_connected(dx, dy, dz):
	return ((dx**2 + dy**2 + dz**2) == 1)

def num_ccs(img):
	[black_labels, num_black_labels] = skim.label(img, neighbors=8, return_num=True)
	[white_labels, num_white_labels] = skim.label(1 - img, neighbors=4, return_num=True)

	return [ num_black_labels, num_white_labels ]

# It is assumed that the border is the same between the two of these
def check_topology(before, after):

	[black_labels_before, num_black_labels_before] = skim.label(before, neighbors=8, return_num=True)
	[white_labels_before, num_white_labels_before] = skim.label(1 - before, neighbors=4, return_num=True)

	[black_labels_after, num_black_labels_after] = skim.label(after, neighbors=8, return_num=True)
	[white_labels_after, num_white_labels_after] = skim.label(1 - after, neighbors=4, return_num=True)

	if (not (num_black_labels_after == num_black_labels_before)) or (not (num_white_labels_after == num_white_labels_before)):
		return False

	num_black_labels = num_black_labels_before
	num_white_labels = num_white_labels_before

	for b_row in range(0, num_black_labels):
		row_label = b_row + 1
		row_img = np.equal(black_labels_before, row_label)

		correspondences = 0

		for b_col in range(0, num_black_labels):
			col_label = b_col + 1
			col_img = np.equal(black_labels_after, col_label)

			correspondences += (np.count_nonzero(np.logical_or(row_img, col_img)) > 1)

		if not (correspondences == 1):
			return False

	for w_row in range(0, num_white_labels):
		row_label = w_row + 1
		row_img = np.equal(white_labels_before, row_label)

		correspondences = 0

		for w_col in range(0, num_white_labels):
			col_label = w_col + 1
			col_img = np.equal(white_labels_after, col_label)

			correspondences += (np.count_nonzero(np.logical_or(row_img, col_img)) > 1)

		if not (correspondences == 1):
			return False


	return True




def label(input_img):
	# Assuming 26-connected black and 6-connected white

	# I think this type of padding works... (should simplify logic)
	padded = np.pad(
		input_img,
		((1, 1), (1, 1), (1, 1)),
		'edge')

	labels = np.zeros((padded.shape), dtype=np.int32)

	img_shape = padded.shape

	init_black_counter = img_shape[0] * img_shape[1] * img_shape[2] + 1
	init_white_counter = 1

	black_label_counter = init_black_counter
	white_label_counter = init_white_counter

	black_sets = []
	white_sets = []

	# Pass 1
	for x in range(1, img_shape[0] - 1):
		for y in range(1, img_shape[1] - 1):
			for z in range(1, img_shape[2] - 1):

				img_val = padded[x, y, z]

				if img_val:
					possible_labels = []
					for dx in range(-1, 2):
						for dy in range(-1, 2):
							for dz in range(-1, 2):
								read_label = labels[x + dx, y + dy, z + dz]
								if read_label and padded[x + dx, y + dy, z + dz]:
									possible_labels.append(read_label)

					if len(possible_labels) > 0:
						lowest_label = min(possible_labels)
						labels[x, y, z] = lowest_label
						for l in possible_labels:
							black_sets[l - init_black_counter].add(lowest_label)
					else:
						labels[x, y, z] = black_label_counter
						black_sets.append(set([black_label_counter]))
						black_label_counter += 1
				else:
					possible_labels = []
					for dx in range(-1, 2):
						for dy in range(-1, 2):
							for dz in range(-1, 2):
								read_label = labels[x + dx, y + dy, z + dz]
								if six_connected(dx, dy, dz) and read_label and (padded[x + dx, y + dy, z + dz] == 0):
									possible_labels.append(read_label)

					if len(possible_labels) > 0:
						lowest_label = min(possible_labels)
						# print(possible_labels)
						labels[x, y, z] = lowest_label
						for l in possible_labels:
							# print(l - init_white_counter)
							white_sets[l - init_white_counter].add(lowest_label)
					else:
						labels[x, y, z] = white_label_counter
						white_sets.append(set([white_label_counter]))
						white_label_counter += 1

	# Pass 2
	num_black_labels = black_label_counter - init_black_counter
	num_white_labels = white_label_counter - init_white_counter

	for b in range(0, num_black_labels):
		b_idx = init_black_counter + b

		union_sets = []

		for k in range(0, num_black_labels):
			if b_idx in black_sets[k]:
				union_sets.append(k)

		final_set = black_sets[union_sets[0]]
		black_sets[union_sets[0]] = set()
		for k in range(1, len(union_sets)):
			final_set = final_set.union(black_sets[union_sets[k]])
			black_sets[union_sets[k]] = set()

		black_sets[b] = final_set


	for w in range(0, num_white_labels):
		w_idx = init_white_counter + w

		union_sets = []

		for k in range(0, num_white_labels):
			if w_idx in white_sets[k]:
				union_sets.append(k)

		final_set = white_sets[union_sets[0]]
		white_sets[union_sets[0]] = set()
		for k in range(1, len(union_sets)):
			final_set = final_set.union(white_sets[union_sets[k]])
			white_sets[union_sets[k]] = set()

		white_sets[w] = final_set

	num_black_ccs = 0
	num_white_ccs = 0

	for i in range(0, len(black_sets)):
		num_black_ccs += (len(black_sets[i]) > 0)

	for i in range(0, len(white_sets)):
		num_white_ccs += (len(white_sets[i]) > 0)

	return [num_black_ccs, num_white_ccs]




def two_pass(input_img):
	# Assuming 26-connected black and 6-connected white

	# I think this type of padding works... (should simplify logic)
	padded = np.pad(
		input_img,
		((1, 1), (1, 1), (1, 1)),
		'edge')

	labels = np.zeros((padded.shape), dtype=np.int32)

	img_shape = padded.shape

	init_black_counter = img_shape[0] * img_shape[1] * img_shape[2] + 1
	init_white_counter = 1

	black_label_counter = init_black_counter
	white_label_counter = init_white_counter

	black_sets = []
	white_sets = []

	# Pass 1
	for x in range(1, img_shape[0] - 1):
		for y in range(1, img_shape[1] - 1):
			for z in range(1, img_shape[2] - 1):

				img_val = padded[x, y, z]

				if img_val:
					possible_labels = []
					for dx in range(-1, 2):
						for dy in range(-1, 2):
							for dz in range(-1, 2):
								read_label = labels[x + dx, y + dy, z + dz]
								if read_label and padded[x + dx, y + dy, z + dz]:
									possible_labels.append(read_label)

					if len(possible_labels) > 0:
						lowest_label = min(possible_labels)
						labels[x, y, z] = lowest_label
						for l in possible_labels:
							black_sets[l - init_black_counter].add(lowest_label)
					else:
						labels[x, y, z] = black_label_counter
						black_sets.append(set([black_label_counter]))
						black_label_counter += 1
				else:
					possible_labels = []
					for dx in range(-1, 2):
						for dy in range(-1, 2):
							for dz in range(-1, 2):
								read_label = labels[x + dx, y + dy, z + dz]
								if six_connected(dx, dy, dz) and read_label and (padded[x + dx, y + dy, z + dz] == 0):
									possible_labels.append(read_label)

					if len(possible_labels) > 0:
						lowest_label = min(possible_labels)
						# print(possible_labels)
						labels[x, y, z] = lowest_label
						for l in possible_labels:
							# print(l - init_white_counter)
							white_sets[l - init_white_counter].add(lowest_label)
					else:
						labels[x, y, z] = white_label_counter
						white_sets.append(set([white_label_counter]))
						white_label_counter += 1

	# Pass 2
	num_black_labels = black_label_counter - init_black_counter
	num_white_labels = white_label_counter - init_white_counter

	for b in range(0, num_black_labels):
		b_idx = init_black_counter + b

		union_sets = []

		for k in range(0, num_black_labels):
			if b_idx in black_sets[k]:
				union_sets.append(k)

		final_set = black_sets[union_sets[0]]
		black_sets[union_sets[0]] = set()
		for k in range(1, len(union_sets)):
			final_set = final_set.union(black_sets[union_sets[k]])
			black_sets[union_sets[k]] = set()

		black_sets[b] = final_set


	for w in range(0, num_white_labels):
		w_idx = init_white_counter + w

		union_sets = []

		for k in range(0, num_white_labels):
			if w_idx in white_sets[k]:
				union_sets.append(k)

		final_set = white_sets[union_sets[0]]
		white_sets[union_sets[0]] = set()
		for k in range(1, len(union_sets)):
			final_set = final_set.union(white_sets[union_sets[k]])
			white_sets[union_sets[k]] = set()

		white_sets[w] = final_set

	num_black_ccs = 0
	num_white_ccs = 0

	for i in range(0, len(black_sets)):
		num_black_ccs += (len(black_sets[i]) > 0)

	for i in range(0, len(white_sets)):
		num_white_ccs += (len(white_sets[i]) > 0)

	return [num_black_ccs, num_white_ccs]


# num = 2000
# start = time.time()
# for n in range(0, num):
# 	test = np.random.random((5, 5, 7)) > 0.5
# 	two_pass(test)
# elapsed = time.time() - start

# average_time = elapsed / num
# print("Average time = " + str(average_time) + " and total time = " + str(elapsed))


# np.random.seed(989187)
# test = np.random.random((5, 5, 5)) > 0.5
# # test = np.zeros((5, 5, 5))
# num_cc = two_pass(test)
# adict = {}
# adict['test'] = test
# adict['num_cc'] = num_cc
# sio.savemat('topo_test.mat', adict)

# print(num_cc)


