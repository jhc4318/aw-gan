import matplotlib.pyplot as plt
from skimage import feature, color, io
import nibabel as nib
from os.path import exists

# Load image
if not exists("image_gt.png"):
    full_image = nib.load("1003_3x1110_3Warped.nii.gz").get_data()
    image = full_image[:,:,99]
    plt.imsave("image_gt.png", image, cmap="gray")

image = io.imread("image_gt.png")
edges = feature.canny(color.rgb2gray(image), sigma=2)
image[edges == True] = [255, 0, 0, 255]
plt.imsave("image_edges.png", image)


