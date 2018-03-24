import os

# Unfortunately, I don't have enough RAM on my gpu to train the model
# If you want to use gpu for other process, such as detection, just comment this line out
# You can also comment this line out if you are using cpu version of tensorflow, or if you have enough gpu memory
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage, skimage.io, skimage.color
from skimage.morphology import label
import pandas as pd

from config import Config
import utils
import model as modellib
import visualize
from model import log

# Establish Random Seed
seed = 42
random.seed = seed
np.random.seed = seed

# Root directory of the project
print(os.getcwd())
ROOT_DIR = os.path.dirname(os.getcwd())

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "output")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# This is a configuration class for training
class MyConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "my_dataset"

    # Train on 1 GPU and 8 images per GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nucleus

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 160
    IMAGE_MAX_DIM = 832

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 400

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 25

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


# The subclass of Dataset which loads the data and creates validation and training data split
# I intend to add data augmentation techniques as well (Hopefully)
class MyDataset(utils.Dataset):

    def load_dataset(self, directory, image_ids=None):
        # Should only call this once, before calling prepare
        self.__init__()
        self.dir = directory
        if image_ids is None:
            image_ids = next(os.walk(directory))[1]

        for id in image_ids:
            self.add_image('Data_Science_Bowl_2018', id, self.dir + id + '/images/' + id + '.png')
            self.image_ids.append(id)

        self.add_class('Data_Science_Bowl_2018', 1, 'Nucleus')

    def randomize_dataset(self):
        self.image_ids = np.random.permutation(self.image_ids)

    def get_train_and_val_datasets(self, split=0.9):
        cutoff = int(split * len(self.image_ids))
        train_data = MyDataset()
        train_data.load_dataset(self.dir, self.image_ids[:cutoff])
        val_data = MyDataset()
        val_data.load_dataset(self.dir, self.image_ids[cutoff:])
        return train_data, val_data

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = image[:,:,:3]
        return image

    def load_mask(self, image_id):
        # info = self.image_info[image_id]
        id = self.image_info[image_id]['id']
        folder = self.dir + id + '/masks/'
        mask = None
        for mask_file in next(os.walk(folder))[2]:
            mask_ = np.array(skimage.io.imread(folder + mask_file), dtype=np.bool)
            mask_ = mask_[:, :, np.newaxis]
            if mask is None:
                mask = mask_
            else:
                mask = np.append(mask, mask_, axis=2)

        classes = np.ones([mask.shape[2]], dtype=np.int32)
        return mask, classes


config = MyConfig()
config.display()

dataset = MyDataset()
dataset.load_dataset('../input/stage1_train/')
mask, classes = dataset.load_mask(10)
print(mask.shape)
print(classes.shape)
print(dataset.image_ids)

dataset.randomize_dataset()
train_data, val_data = dataset.get_train_and_val_datasets()

train_data.prepare()
val_data.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "other"  # imagenet, coco, last, or other

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
elif init_with == "other":
    model_path = os.path.join(ROOT_DIR, "mask_rcnn_test1.h5")
    model.load_weights(model_path, by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(train_data, val_data,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(train_data, val_data,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10,
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_v2-0.h5")
model.keras_model.save_weights(model_path)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class InferenceConfig(MyConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, "output/mask_rcnn_test1.h5")
# model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(val_data.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(val_data, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            train_data.class_names, figsize=(8, 8))


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            val_data.class_names, r['scores'], ax=get_ax())

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(val_data.image_ids, 10)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#         modellib.load_image_gt(val_data, inference_config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps = \
#         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)
#
# print("mAP: ", np.mean(APs))


# Here we are going to try and do the run-length encoding
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


# Create and prepare test data set
test_data = MyDataset()
test_data.load_dataset('../input/stage1_test/')
test_data.prepare()

print('The number of image_ids is: ', len(test_data.image_ids))

new_test_ids = []
rles = []
# Go through each image id and run detection
for id in test_data.image_ids:
    image = test_data.load_image(id)
    results = model.detect([image])
    r = results[0]
    rle_list = []
    id_list = []
    masks = r['masks']
    scores = r['scores']
    classes = r['class_ids']

    for i in range(len(scores)):
        m = masks[:, :, i]
        s = scores[i]
        c = classes[i]
        if c == 1 and s > 0.5:
            rle_list.append(rle_encoding(m))
            id_list.append(train_data.image_info[id]['id'])
        else:
            print('Mask not accepted as score is: ', s)

    if len(rle_list) == 0:
        print('This image id was not included for some reason: ', train_data.image_info[id]['id'])

    new_test_ids.extend(id_list)
    rles.extend(rle_list)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-2.0.csv', index=False)

