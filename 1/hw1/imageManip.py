import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    
    ### YOUR CODE HERE
    dim = lambda xp: 0.5*(xp**2) ##applies the dim filter to the image
    
    #dimension variable is only use for extracting the data in one line
    rows,columns,dimension = image.shape
    
    #iterate over pixels
    image1 = image.copy()
    for row in range(rows):
        for column in range(columns):
            #apply dim filter to each color and combine the result
            r = dim(image1[column,row][0])
            g = dim(image1[column,row][1])
            b = dim(image1[column,row][2])
            image1[column,row] = (r, g, b)
    out = image1
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    if not channel in "RGB":
        return print("The channel you have choosen is not valid (▀̿Ĺ̯▀̿ ̿)")
    
    out = None
    ### YOUR CODE HERE
    # in order to exclude the channels I create a copy of the image and apply the respective
    # filtes, assigning a channel to 0 excludes it from the image
    if channel == "R":
        exred_image = image.copy()
        exred_image[:,:,0] = 0 #red
        out = exred_image
    elif channel == "G":
        exgreen_image = image.copy() 
        exgreen_image[:,:,1] = 0 #green
        out = exgreen_image
    elif channel == "B":
        exblue_image = image.copy()
        exblue_image[:,:,2] = 0 #blue
        out=exblue_image
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image.copy())
    out = None
    ### YOUR CODE HERE
    if channel == 'H':
        hue_img = hsv[:, :, 0]
        out = hue_img
    elif channel == 'S':
        sat_img = hsv[:, :, 1]
        out = sat_img
    elif channel == 'V':
        value_img = hsv[:, :, 2]
        out = value_img
    ### END YOUR CODE
    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    s1 = rgb_exclusion(image1.copy(), channel1)
    s2 = rgb_exclusion(image2.copy(), channel2)
    #get image dimensions
    height1, width1,x = s1.shape
    height2, width2,x = s2.shape
    #cutting image in half
    width_cutoff1 = width1 // 2
    width_cutoff2 = width2 // 2
    left = s1[:, :width_cutoff1]
    right = s2[:, width_cutoff2:]
    
    out = np.concatenate((left, right), axis=1)
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
