# import numpy as np
# import cv2 as cv
# from time import sleep
# import matplotlib.pyplot as plt
# import math

# def gaus_blur(size, sigma=1):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage import data

# lena = sp.misc.lena() this function was deprecated in version 0.17
img = data.camera()  # use a standard image from skimage instead
LoG = nd.gaussian_laplace(img, 2)
thres = np.absolute(LoG).mean() * 0.75
output = sp.zeros(LoG.shape)
w = output.shape[1]
h = output.shape[0]

for y in range(1, h - 1):
    for x in range(1, w - 1):
        patch = LoG[y-1:y+2, x-1:x+2]
        p = LoG[y, x]
        maxP = patch.max()
        minP = patch.min()
        if (p > 0):
            zeroCross = True if minP < 0 else False
        else:
            zeroCross = True if maxP > 0 else False
        if ((maxP - minP) > thres) and zeroCross:
            output[y, x] = 1

plt.imshow(output)
plt.show()
# def l_o_g(x, y, sigma):
#     # Formatted this way for readability
#     nom = ((y**2)+(x**2)-2*(sigma**2))
#     denom = ((2*math.pi*(sigma**6)))
#     expo = math.exp(-((x**2)+(y**2))/(2*(sigma**2)))
#     return nom*expo/denom


# def create_log(sigma, size=7):
#     w = math.ceil(float(size)*float(sigma))

#     if (w % 2 == 0):
#         w = w + 1

#     l_o_g_mask = []

#     w_range = int(math.floor(w/2))
#     print("Going from " + str(-w_range) + " to " + str(w_range))
#     for i in range(-w_range, w_range):
#         for j in range(-w_range, w_range):
#             l_o_g_mask.append(l_o_g(i, j, sigma))
#     l_o_g_mask = np.array(l_o_g_mask)
#     l_o_g_mask = l_o_g_mask.reshape(w, w)
#     return l_o_g_mask


# def convolve(image, mask):
#     width = image.shape[1]
#     height = image.shape[0]
#     w_range = int(math.floor(mask.shape[0]/2))

#     res_image = np.zeros((height, width))

#     # Iterate over every pixel that can be covered by the mask
#     for i in range(w_range, width-w_range):
#         for j in range(w_range, height-w_range):
#             # Then convolute with the mask
#             for k in range(-w_range, w_range):
#                 for h in range(-w_range, w_range):
#                     res_image[j, i] += mask[w_range +
#                                             h, w_range+k]*image[j+h, i+k]
#     return res_image


# def z_c_test(l_o_g_image):
#     z_c_image = np.zeros(l_o_g_image.shape)

#     # Check the sign (negative or positive) of all the pixels around each pixel
#     for i in range(1, l_o_g_image.shape[0]-1):
#         for j in range(1, l_o_g_image.shape[1]-1):
#             neg_count = 0
#             pos_count = 0
#             for a in range(-1, 1):
#                 for b in range(-1, 1):
#                     if (a != 0 and b != 0):
#                         if (l_o_g_image[i+a, j+b] < 0):
#                             neg_count += 1
#                         elif (l_o_g_image[i+a, j+b] > 0):
#                             pos_count += 1

#             # If all the signs around the pixel are the same and they're not all zero, then it's not a zero crossing and not an edge.
#             # Otherwise, copy it to the edge map.
#             z_c = ((neg_count > 0) and (pos_count > 0))
#             if (z_c):
#                 z_c_image[i, j] = 1

#     return z_c_image


# def run_l_o_g(bin_image, sigma_val, size_val):
#     # Create the l_o_g mask
#     print("creating mask")
#     l_o_g_mask = create_log(sigma_val, size_val)

#     # Smooth the image by convolving with the LoG mask
#     print("smoothing")
#     l_o_g_image = convolve(bin_image, l_o_g_mask)

#     # Display the smoothed imgage
#     blurred = plt.add_subplot(1, 4, 2)
#     # blurred.imshow(l_o_g_image, cmap=cm.gray)

#     # Find the zero crossings
#     print("finding zero crossings")
#     z_c_image = z_c_test(l_o_g_image)
#     print(z_c_image)

#     # Display the zero crossings
#     edges = plt.add_subplot(1, 4, 3)
#     # edges.imshow(z_c_image, cmap=cm.gray)
#     # pylab.show()


# with opencv2
# image = cv.imread(
#     "/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/mosaic_BKP.jpg")

# run_l_o_g(image, 1, 400)
# # height, width, channels = image.shape

# # print(l_o_g(width, height, 1))

# # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# # blur_image = cv.GaussianBlur(gray_image, (3, 3), 0)
# # laplacian = cv.Laplacian(blur_image, cv.CV_64F)

# # plt.figure()
# # plt.title('Shapes')
# # plt.imsave('shapes-lap.png', laplacian, cmap='gray', format='png')
# # plt.imshow(laplacian, cmap='gray')

# plt.show()
# # Calibrate and install kinect
