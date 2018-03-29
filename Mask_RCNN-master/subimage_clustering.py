import random
import numpy as np
import scipy as sp
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
k_vals = (16, 32, 64, 128)
output_dir = '../output' + image_type
output_filenames = ('16_means_clusters.png', '32_means_clusters.png'    , '64_means_clusters.png', '128_means_clusters.png')

# Set image variables
title = 'LAB Color Cluster Centers w/o  Normalization'
subplot_x = (4, 4, 8, 8)
subplot_y = (4, 8, 8, 16)
conversion_type = 'lab2rgb'

train_ids = next(os.walk(train_dir))[1]
# test_ids = next(os.walk('../input/stage1_test/'))[1]

# test to see if the lab2rgb is working
# for i in range(100):
#     print(i)
#     orig_image = np.array(skimage.io.imread(train_dir + train_ids[i] + '/images/' + train_ids[i] + '.png'))
#     plt.imshow(orig_image)
#     plt.show()
#     lab_image = np.array(skimage.io.imread(train_dir + train_ids[i] + image_type + train_ids[i] + '.png'))
#     plt.imshow(lab_image)
#     plt.show()
#     rgb_image = skimage.color.lab2rgb(skimage.color.rgb2lab(lab_image))
#     plt.imshow(rgb_image)
#     plt.show()

image_ids = []
heights = []
widths = []
pixels = []

for image_id in train_ids:
    image_ids.append(image_id)
    image = np.array(skimage.io.imread(train_dir + image_id + image_type + image_id + '.png'))
    heights.append(image.shape[0])
    widths.append(image.shape[1])
    pixels.append(image.shape[0] * image.shape[1])

probabilities = np.zeros(len(image_ids))
total_pixels = sum(pixels)
current_prob = 0
for i in range(len(image_ids)):
    probabilities[i] = current_prob + (pixels[i] / total_pixels)
    current_prob = probabilities[i]

print(probabilities[-1])

def get_rand_image(dir, image_ids, probabilities):
    r = random.random()
    j = 0
    while r > probabilities[j]:
        j += 1
        if j >= len(probabilities):
            break

    if j >= len(probabilities):
        return get_rand_image(dir, image_ids, probabilities)
    else:
        image_id = image_ids[j]
        image = np.array(skimage.io.imread(dir + image_id + image_type + image_id + '.png'))
        if conversion_type == 'lab2rgb':
            return skimage.color.rgb2lab(image)
        else:
            return image

def get_rand_sub_image(image, sub_height, sub_width):
    pad_h = int((sub_height - 1) / 2)
    pad_w = int((sub_width - 1) / 2)
    y = random.randint(0 + pad_h, image.shape[0] - 1 - pad_h)
    x = random.randint(0 + pad_w, image.shape[1] - 1 - pad_w)
    if IMG_CHANNELS == 1:
        image = np.expand_dims(image, 2)

    pad_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=0)
    test = pad_image[y:y + 2 * pad_h + 1, x:x + 2 * pad_w + 1, :IMG_CHANNELS]
    if test.shape != (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        print(x, y, pad_w, pad_h)
        print(image.shape)
        print(pad_image.shape)

    return pad_image[y:y + 2*pad_h + 1, x:x + 2*pad_w + 1, :IMG_CHANNELS]


vectors = []
for i in range(n_samples):
    if i % 100 == 0:
        print(i)
    sub_image = get_rand_sub_image(get_rand_image(train_dir, image_ids, probabilities), IMG_HEIGHT, IMG_WIDTH)
    vectors.append(np.reshape(sub_image, (IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS)))

# Do k-means clustering with several values of k
for k_try, rows, cols, output_file in zip(k_vals, subplot_x, subplot_y, output_filenames):
    (k_means_centroids, k_means_labels, k_means_cost) = clust.k_means(vectors, k_try, verbose=1)

    print(k_means_cost)
    fig, subplots = plt.subplots(rows, cols)
    subplots = np.reshape(subplots, (k_try))
    for i in range(len(subplots)):
        if IMG_CHANNELS == 1:
            center_image = np.reshape(k_means_centroids[i], (IMG_HEIGHT, IMG_WIDTH)).astype(np.uint8)
            subplots[i].imshow(center_image, cmap='gray')
        else:
            if conversion_type == 'lab2rgb':
                center_image = np.reshape(k_means_centroids[i], (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                center_image = skimage.color.lab2rgb(center_image)
            else:
                center_image = np.reshape(k_means_centroids[i], (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)).astype(np.uint8)

            subplots[i].imshow(center_image)

    fig.suptitle(title + ' k=' + str(k_try))
    plt.show(block=False)
    plt.savefig(output_dir + output_file)


# (mean_shift_centers, mean_shift_labels) = clust.mean_shift(np.array(vectors))
# print(len(mean_shift_centers))
#
# fig, subplots = plt.subplots(8, 8)
# subplots = np.reshape(subplots, (64))
# for i in range(len(mean_shift_centers)):
#     center_image = np.reshape(mean_shift_centers[i], (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)).astype(np.uint8)
#     subplots[i].imshow(center_image)
#
# plt.show()


