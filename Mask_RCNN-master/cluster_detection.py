import random
import math
import numpy as np
import pandas as pd
import os
from skimage.morphology import label
import skimage.io, skimage.color

print(os.getcwd())

# Set all of the variable needed
test_dir = '../input/stage1_test/'
image_type = '/lab_norm/'
IMG_HEIGHT = 21
IMG_WIDTH = 21
IMG_CHANNELS = 1
k_vals = 128# (16, 32, 64, 128)
input_dir = '../output/lab_norm/'
center_dir = input_dir + 'Centers/'
masks_dir = input_dir + 'Center_masks/'
output_dir = '../output/lab_norm/'

# train_ids = next(os.walk(train_dir))[1]
test_ids = next(os.walk(test_dir))[1]


# Detection Algorithm
def detect_image(image, centers, masks):
    h = image.shape[0]
    w = image.shape[1]
    i = 0
    j = 0
    image_mask = np.zeros((image.shape[0], image.shape[1]))
    while i < h:
        if i > h - IMG_HEIGHT:
            i = h - IMG_HEIGHT
        while j < w:
            if j > w - IMG_WIDTH:
                j = w - IMG_WIDTH
            sub_image = image[i:i+IMG_HEIGHT,j:j+IMG_WIDTH]
            index = -1
            min_dist = math.inf
            for k in range(len(centers)):
                temp_dist = np.linalg.norm(sub_image.flatten() - centers[k].flatten())
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    index = k

            image_mask[i:i+IMG_HEIGHT,j:j+IMG_WIDTH] = masks[index]
            j += IMG_WIDTH

        i += IMG_HEIGHT
        j = 0

    return image_mask



# Load in the cluster centers and masks
center_filenames = next(os.walk(center_dir))[2]
print(len(center_filenames))
center_hashmap = [''] * len(center_filenames)
for cf in center_filenames:
    id = cf.split('_')[1]
    id = int(id.split('.')[0])
    center_hashmap[id] = cf

mask_filenames = next(os.walk(masks_dir))[2]
print(len(mask_filenames))
mask_hashmap = [''] * len(mask_filenames)
for cf in mask_filenames:
    id = cf.split('_')[1]
    id = int(id.split('.')[0])
    mask_hashmap[id] = cf

centers = []
center_masks = []
for cf, mf in zip(center_hashmap, mask_hashmap):
    if IMG_CHANNELS != 1:
        img = np.array(skimage.io.imread(center_dir + cf))
        centers.append(img[:, :, 0:IMG_CHANNELS])
    else:
        centers.append(np.array(skimage.io.imread(center_dir + cf, as_grey=True)))
    center_masks.append(np.array(skimage.io.imread(masks_dir + mf, as_grey=True)))

# Load in one image at a time and detect
image_ids = []
images = []
masks = []

count = 0
for image_id in test_ids:
    count += 1
    print(count)
    image_ids.append(image_id)
    image = np.array(skimage.io.imread(test_dir + image_id + image_type + image_id + '.png'))
    if IMG_CHANNELS != 1:
        image = image[:, :, 0:IMG_CHANNELS]
    mask = detect_image(image, centers, center_masks)
    images.append(image)
    masks.append(mask)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


new_test_ids = []
rles = []
print('Beginning RLE')
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(masks[n]))
    if len(rle) == 0:
        print(n)
        print(id_)
        print(np.where(masks[n]))
        rle = [[1, 1]]
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(output_dir + 'sub-dsbowl2018-clustering-lab_gray.csv', index=False)
