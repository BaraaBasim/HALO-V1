from __future__ import division
from os import mkdir;
from PIL import Image, ImageOps
import array
# import matlab.engine
import numpy as np
import imageio.v2 as iio
import cv2
import matplotlib.pyplot as plt

# eng = matlab.engine.start_matlab();
# def generatePSF(eng, top, bottom, left, right, r, g, b):
#     # Generate the PSF
#     img = eng.psf(r, g, b)
#     img = np.array(img)
#     img = (img / 65535.0 * 255.0).astype(np.uint8)

#     # Add padding to the image
#     top = int(top)
#     bottom = int(bottom)
#     left = int(left)
#     right = int(right)
#     padded_img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant')

#     return padded_img

# Modified version to work with Jpg images 
def add_padding(img, top, bottom, left, right):
    img = np.array(img)
    top = int(top)
    bottom = int(bottom)
    left = int(left)
    right = int(right)
    # print('top: ', top,' bottom: ', bottom, ' left: ', left, ' right: ' ,right)
    padded_img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant')
    return padded_img

# def add_light_source(img, light_source, padding_values):
#     top, bottom, left, right = padding_values
#     img_height, img_width = img.shape[:2]
#     ls_height, ls_width = light_source.shape[:2]

#     # Resize the light source to a smaller size
#     light_source = cv2.resize(light_source, (0,0), fx=0.05, fy=0.05)

#     # Add padding to the light source
#     padded_light_source = cv2.copyMakeBorder(light_source, top, bottom, left, right, cv2.BORDER_CONSTANT, dst=None, value=[0,0,0])
    
#     # Calculate the scale for each side
#     padded_ls_height, padded_ls_width = padded_light_source.shape[:2]
#     scale_top = img_height / padded_ls_height
#     scale_left = img_width / padded_ls_width

#     # Resize the padded light source image using the calculated scales
#     padded_light_source = cv2.resize(padded_light_source, (0,0), fx=scale_left, fy=scale_top)
#     print("size of the light suorce after the padding: ",padded_light_source.size)
#     print("size of the target: ",img.size)
#     # Set the brightness and Contrast values
#     brightness = 50
#     contrast = 50

#     # Apply the brightness and contrast to the padded light source image
#     cv2.addWeighted(padded_light_source, 1 + contrast/127, np.zeros(padded_light_source.shape, dtype=padded_light_source.dtype), 0, brightness - contrast, padded_light_source)
#     # Add the adjusted light source image to the input image 
#     cv2.addWeighted(img, 1, padded_light_source, 1, 0, img)
#     return img

# image_no = 1
# filename = "attack_pipeline/psf_starlight" + str(image_no) + '.jpg'
# img = generatePSF(eng, 0, 0, 0, 0, 255, 0, 0);
# plt.imsave(filename, img)
# eng.quit()
