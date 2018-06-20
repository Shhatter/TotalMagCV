import os

import cv2
import imutils
import matplotlib.pyplot as plt
import imageio
from skimage.feature import hog
from skimage import data, exposure, color, data_dir, img_as_uint, img_as_ubyte, img_as_int
import numpy as np
from skimage.filters.rank import equalize
from skimage.morphology import disk
from skimage.viewer import ImageViewer

image = color.rgb2gray(data.astronaut())
correct_output = np.load(
    os.path.join(data_dir, 'astronaut_GRAY_hog_L1.npy'))

fd, hog_image = hog(image=image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys',
                    visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
hogrez = imutils.resize(hog_image, 1000)
img_rescale = exposure.equalize_hist(hogrez)
cv2.imshow("global equalize", img_rescale)
hogrez = img_as_uint(hogrez)
# cv2.imshow("HOG", hogrez)
ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

ax2.save
# plt.show()
'''

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

# hoggyGray = cv2.cvtColor(hog_image, cv2.COLOR_BGR2GRAY)
# hoggy = cv2.applyColorMap(hog_image,cv2.COLORMAP)
# hoggy = img_as_uint(hog_image)
# hoggy = imutils.resize(hoggy, 1000)


p2, p98 = np.percentile(hog_image, (2, 98))
# img_rescale = exposure.rescale_intensity(hog_image, in_range=(p2, p98))
img_rescale = exposure.equalize_adapthist(hog_image, clip_limit=0.03)

img_rescale = imutils.resize(img_rescale, 1000)
cv2.imshow('asdasdasdad', img_rescale)
cv2.waitKey(0)

# cv2.imshow('asdasdasdad', hoggyGray)
# cv2.waitKey(0)


# cv2.imshow("hoggy ", hoggy)
# cv2.waitKey(0)
clahe = cv2.createCLAHE(clipLimit=11.0, tileGridSize=(16, 16))
cl1 = clahe.apply(hoggy)
cv2.imshow('clahe_2.jpg', cl1)

plt.show()

selem = disk(8)
img_eq = equalize(hoggy, selem=selem)
cv2.imshow("local equalize", img_eq)
cv2.waitKey(0)

'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

'''
