import numpy as np
from scipy import ndimage
from PIL import Image
import math


def gaussian_kernel(size, sigma):
    """Returns a 2D Gaussian kernel."""
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def LoG_kernel(size, sigma):
    """Returns a Laplacian of Gaussian (LoG) kernel."""
    x, y = np.mgrid[-size:size+1, -size:size+1]
    nom = ((y**2)+(x**2)-2*(sigma**2))
    denom = ((2*np.pi*(sigma**6)))
    expo = np.exp(-((x**2)+(y**2))/(2*(sigma**2)))
    return nom*expo/denom


def convolve(image, kernel):
    """Convolves an image with a kernel."""
    return ndimage.convolve(image, kernel)


def LoG(image, size, sigma):
    """Applies Laplacian of Gaussian edge detection to an image."""
    kernel = LoG_kernel(size, sigma)
    convolved = convolve(image, kernel)
    return convolved


# Load an image and convert it to grayscale
img = np.array(Image.open(
    '/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/mosaic.jpg').convert('L'))

# Apply Laplacian of Gaussian edge detection with a 5x5 kernel and sigma=1.4
edges = LoG(img, 5, 1.4)

# Save the result to a file
Image.fromarray(edges).save('output.jpg')
