import random
import numpy as np
import os
import skimage.io, skimage.color
import sklearn.cluster as clust
import matplotlib.pyplot as plt

print(os.getcwd())

# Set all of the variable needed
train_dir = '../input/stage1_train/'
image_type = '/lab_color/'
n_samples = 50000
IMG_HEIGHT = 21
IMG_WIDTH = 21
IMG_CHANNELS = 3
k_vals = (64)# (16, 32, 64, 128)
output_dir = '../output' + image_type
output_filenames = ('64_means_clusters.png') # ('16_means_clusters.png', '32_means_clusters.png', '64_means_clusters.png', '128_means_clusters.png')
center_dir = output_dir + 'Centers/'
masks_dir = output_dir + 'Center_masks/'

# Set image variables
title = 'RBG Cluster Centers w/o  Normalization'
subplot_x = (8) # (4, 4, 8, 8)
subplot_y = (8) # (4, 8, 8, 16)
conversion_type = 'none'

# train_ids = next(os.walk(train_dir))[1]
test_ids = next(os.walk('../input/stage1_test/'))[1]
