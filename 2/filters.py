from matplotlib.colors import from_levels_and_colors
import numpy as np
from numpy.ma.core import get_object_signature


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            ##kernel axis
            addition = 0.0
            for i in range(Hk):
                for j in range(Wk):
                    if (m + 1 - i) < 0 or (n + 1 - j) < 0 or (
                            m + 1 - i) >= Hi or (n + 1 - j) >= Wi:
                        addition += 0
                    else:
                        addition += kernel[i][j] * image[m + 1 - i][n + 1 - j]
            out[m][n] = addition

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    out = None

    ### YOUR CODE HERE
    #creating mold
    out = np.pad(image, [(pad_height, ), (pad_width, )], mode='constant')
    #baking cake

    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    kernelflip = np.flipud(np.fliplr(kernel))  #easy way to flip
    imagePadded = zero_pad(image, Hk // 2, Wk // 2)
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            out[m][n] = np.sum(imagePadded[m:m + Hk, n:n + Wk] *
                               kernelflip)  #applying convolution

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    kernel = np.flipud(np.fliplr(g))
    out = conv_fast(f, kernel)
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    gMean = np.mean(g, axis=(0, 1))
    gZeroMean = g - gMean

    out = cross_correlation(f, gZeroMean)

    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    """     
    
    # I tried to keep the code simpler using this code below, but i was not getting the right
    # answer since I need to normailze each sub image but I dont know why it doesnt work
    fmean = np.mean(f)
    gmean = np.mean(g)
    fstd = np.std(f)
    gstd = np.std(g)

    fout = (f - fmean) / fstd
    gout = (g - gmean) / gstd
    out = cross_correlation(fout,gout) 

    """
    # dimensions
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    #padding image
    imagePadded = zero_pad(f, Hk // 2, Wk // 2)
    #buffering output
    out = np.zeros((Hi, Wi))
    #calculating template normalize values
    gmean = np.mean(g)
    gstd = np.std(g)
    gout = (g - gmean) / gstd
    #cross correlation
    for m in range(Hi):
        for n in range(Wi):
            #normalizing subimage of f
            subimage = imagePadded[m:m + Hk, n:n + Wk]
            fmean = np.mean(subimage)
            fstd = np.std(subimage)
            fout = (subimage - fmean) / fstd
            #weighted sum
            out[m][n] = np.sum(fout * gout)
    ### END YOUR CODE
    return out
