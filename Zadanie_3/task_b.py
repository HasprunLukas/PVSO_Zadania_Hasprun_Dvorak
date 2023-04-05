import numpy as np
from PIL import Image
import math
import cv2


def LoG_kernel(size, sigma):
    # https: // homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    x, y = np.meshgrid(np.arange(-(size-1)//2, (size-1)//2+1),
                       np.arange(-(size-1)//2, (size-1)//2+1))
    frac1 = (-1/(math.pi*sigma**4))
    frac2 = (1 - ((x**2+y**2)/(2*sigma**2)))
    expo = np.exp(-((x**2)+(y**2))/(2*(sigma**2)))

    return frac1*frac2*expo


def convolve(image, kernel):
    img_shape = image.shape
    kernel_shape = kernel.shape
    # Compute the output shape
    output_shape = (img_shape[0] - kernel_shape[0] + 1,
                    img_shape[1] - kernel_shape[1] + 1)
    # Initialize the output image
    output_image = np.zeros(output_shape)
    # Perform the convolution
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output_image[i, j] = (
                image[i:i + kernel_shape[0], j:j + kernel_shape[1]] * kernel).sum()
    return output_image


def LoG(image, size, sigma):
    kernel = LoG_kernel(size, sigma)
    convolved_array = convolve(image, kernel)
    return convolved_array


# Load an image and convert it to grayscale
img = np.array(Image.open(
    '/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/chessboard0.jpg').convert('L'))

output_arr = LoG(img, 7, 0.5)
image = cv2.imread(
    '/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/chessboard0.jpg')

ddepth = cv2.CV_16S
kernel_size = 3
src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)

abs_dst = cv2.convertScaleAbs(dst)
cv2.imshow('Opencv Edge detection', abs_dst)
cv2.imshow('From Scratch Edge detection', output_arr)
cv2.waitKey(0)
