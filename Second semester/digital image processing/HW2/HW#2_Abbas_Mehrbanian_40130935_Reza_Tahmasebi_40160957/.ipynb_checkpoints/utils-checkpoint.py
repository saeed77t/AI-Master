import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_gray_img(path, new_size=None):
    if(new_size):
        return cv2.resize(cv2.imread(path, 0), (new_size[0], new_size[1]), interpolation=cv2.INTER_AREA)
    else:
        return cv2.imread(path, 0)


def read_rgb_img(path, new_size=None):
    if(new_size):
        return cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB),
                          (new_size[0], new_size[1]), interpolation=cv2.INTER_AREA)
    else:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def show_img(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    frame1 = fig.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    ax.set_title(title)
    plt.show()

def normalize_image(image):
    """
    Normalizes the given image by scaling the pixel values to be between 0 and 1.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The normalized image as a NumPy array.
    """
    return np.divide(image, 255)

def abs(x):
    """
    Computes the absolute value of a complex NumPy array.
    
    :param x: Input array as a complex NumPy array.
    :return: Array containing the absolute values of the input array.
    """
    
    # Create an empty array to store the absolute values
    result = np.zeros(x.shape)
    
    # Loop over each element in the input array
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i,j] = np.sqrt(x[i,j].real**2 + x[i,j].imag**2)
    
    return result