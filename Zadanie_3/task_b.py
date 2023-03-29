import numpy as np
from PIL import Image


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


# Load an image and convert it to grayscale
img = np.array(Image.open(
    '/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/mosaic_BKP.jpg').convert('L'))

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
output_arr = convolve(img, kernel)

output_img = Image.fromarray(output_arr).convert('L')
output_img.save('output_image.jpg')
