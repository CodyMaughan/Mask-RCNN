import random
import numpy as np
import os
import skimage.io, skimage.color
import sklearn.cluster as clust
import matplotlib.pyplot as plt

print(os.getcwd())

# Set all of the variable needed
train_dir = '../input/stage1_train/'
image_type = '/lab_norm/'
n_samples = 50000
IMG_HEIGHT = 21
IMG_WIDTH = 21
IMG_CHANNELS = 1
k_vals = [64, 128]# (16, 32, 64, 128)
output_dir = '../output/lab_norm/' # '../output' + image_type
output_filenames = ['64_means_clusters.png', '128_means_clusters.png'] # ('16_means_clusters.png', '32_means_clusters.png', '64_means_clusters.png', '128_means_clusters.png')
output_filenames2 = ['64_means_cluster_masks.png', '128_means_cluster_masks.png']

# Set image variables
title_center = 'Grayscale Cluster Centers w/  Normalization'
title_mask = 'Grayscale Cluster Center Masks w/ Normalization'
subplot_x = [8, 8] # (4, 4, 8, 8)
subplot_y = [8, 16] # (4, 8, 8, 16)
conversion_type = 'none'

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
        mask = np.zeros(image.shape[0:2], dtype=np.bool)
        mask_ids = next(os.walk(dir + image_id + '/masks/'))[2]
        for mid in mask_ids:
            m = np.array(skimage.io.imread(dir + image_id + '/masks/' + mid), dtype=np.bool)
            mask = np.bitwise_or(mask, m)

        if conversion_type == 'lab2rgb':
            return skimage.color.rgb2lab(image), mask
        else:
            return image, mask


def get_rand_sub_image(image, sub_height, sub_width):
    mask = image[1]
    image = image[0]
    pad_h = int((sub_height - 1) / 2)
    pad_w = int((sub_width - 1) / 2)
    y = random.randint(0 + pad_h, image.shape[0] - 1 - pad_h)
    x = random.randint(0 + pad_w, image.shape[1] - 1 - pad_w)
    if IMG_CHANNELS == 1:
        image = np.expand_dims(image, 2)

    pad_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=0)
    pad_mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
    test = pad_image[y:y + 2 * pad_h + 1, x:x + 2 * pad_w + 1, :IMG_CHANNELS]
    if test.shape != (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        print(x, y, pad_w, pad_h)
        print(image.shape)
        print(pad_image.shape)

    return pad_image[y:y + 2*pad_h + 1, x:x + 2*pad_w + 1, :IMG_CHANNELS], pad_mask[y:y + 2*pad_h + 1, x:x + 2*pad_w + 1]


vectors = []
masks = []
for i in range(n_samples):
    if i % 100 == 0:
        print(i)
    sub_image, sub_mask = get_rand_sub_image(get_rand_image(train_dir, image_ids, probabilities), IMG_HEIGHT, IMG_WIDTH)
    vectors.append(np.reshape(sub_image, (IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS)))
    masks.append(sub_mask)

masks = np.asarray(masks)
for mask in masks:
    pass

# Do k-means clustering with several values of k
for k_try, rows, cols, output_file, output_file2 in zip(k_vals, subplot_x, subplot_y, output_filenames, output_filenames2):
    orig_images = list(vectors)
    print('beginning clustering with k=' + str(k_try))
    (k_means_centroids, k_means_labels, k_means_cost) = clust.k_means(vectors, k_try, verbose=0)
    print(k_means_cost)
    print(len(k_means_labels))

    center_masks = np.zeros((k_try, IMG_HEIGHT, IMG_WIDTH))
    cluster_counts = [0] * k_try
    for i in range(len(k_means_labels)):
        lab = k_means_labels[i]
        center_masks[lab] = center_masks[lab] + masks[i]
        cluster_counts[lab] += 1

    fig, subplots = plt.subplots(rows, cols)
    fig2, subplots2 = plt.subplots(rows, cols)
    subplots = np.reshape(subplots, k_try)
    subplots2 = np.reshape(subplots2, k_try)
    for i in range(k_try):
        center_mask = ((center_masks[i] / cluster_counts[i]) >= 0.5)

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

        subplots[i].axes.xaxis.set_ticklabels([])
        subplots[i].axes.yaxis.set_ticklabels([])
        subplots2[i].imshow(center_mask, cmap='gray')
        subplots2[i].axes.xaxis.set_ticklabels([])
        subplots2[i].axes.yaxis.set_ticklabels([])
        if IMG_CHANNELS == 1:
            plt.imsave(output_dir + 'Centers/center_' + str(i) + '.png', center_image, cmap=plt.get_cmap('gray'))
        else:
            plt.imsave(output_dir + 'Centers/center_' + str(i) + '.png', center_image)
        plt.imsave(output_dir + 'Center_masks/mask_' + str(i) + '.png', center_mask, cmap=plt.get_cmap('gray'))

    fig.suptitle(title_center + ' k=' + str(k_try))
    fig.savefig(output_dir + output_file)
    fig2.suptitle(title_mask + ' k=' + str(k_try))
    fig2.savefig(output_dir + output_file2)


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


