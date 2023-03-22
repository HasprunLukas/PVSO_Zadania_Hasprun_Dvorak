import numpy as np
from scipy import ndimage
from PIL import Image
import math

def LoG_kernel(size, sigma):
    """Returns a Laplacian of Gaussian (LoG) kernel."""
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    # x, y = np.mgrid[-size:size+1, -size:size+1]
    # x = np.matrix([[0, -1, 0],
    #                [-1, 4, -1],
    #                [0, -1, 0]])
    # y = np.rot90(x)
    x, y = np.meshgrid(np.arange(-(size-1)//2, (size-1)//2+1),
                       np.arange(-(size-1)//2, (size-1)//2+1))
    # x = np.matrix([[0, 0, -1, -1, -1, 0, 0],
    #                [0, -1, -3, -3, -3, -1, 0],
    #                [-1, -3, 0, 7, 0, -3, -1],
    #                [-1, -3, 7, 24, 7, -3, -1],
    #                [-1, -3, 0, 7, 0, -3, -1],
    #                [0, -1, -3, -3, -3, -1, 0],
    #                [0, 0, -1, -1, -1, 0, 0]])
    # y = np.rot90(x)
    frac1 = (-1/(math.pi*sigma**2))
    frac2 = (1 - ((x**2+y**2)/(2*sigma**2)))
    expo = np.exp(-((x**2)+(y**2))/(2*(sigma**2)))

    return frac1*frac2*expo


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
    '../chessboard0.jpg').convert('L'))

# Apply Laplacian of Gaussian edge detection with a 5x5 kernel and sigma=1.4
edges = LoG(img, 5, 1.3)

# Save the result to a file
Image.fromarray(edges).save('output.jpg')
