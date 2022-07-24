import matplotlib.pyplot as plt
from skimage import feature, color, io
import nibabel as nib
from os.path import exists
import tensorflow as tf
import numpy as np
import math

# Load image
if not exists("image_gt.png"):
    full_image = nib.load("1003_3x1110_3Warped.nii.gz").get_data()
    image = full_image[:,:,99]
    plt.imsave("image_gt.png", image, cmap="gray")

image = io.imread("image_gt.png")
image = tf.convert_to_tensor(image)
test = np.array(image)
print(test)
# Canny on numpy array
# Superimpose onto numpy array
# Convert to tensor

plt.imsave("image_edges.png", image)


