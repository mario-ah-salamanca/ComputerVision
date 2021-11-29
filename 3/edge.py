"""
Computer Vision Homework 3
Mario Hernandez
m.hernandez@jacobs-university.de
Copyright 2021
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    ### YOUR CODE HERE
    flippkernel = np.flip(kernel, axis=(0,1))
    for m in range(Hi):
        for n in range (Wi):
            out[m][n] = np.sum(padded[m:m+Hk, n:n+Wk] * flippkernel)
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))
    k = (size -1 )// 2
    ### YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            kernel[i][j] = 1/(2 * np.pi * sigma**2)* np.exp(-1* ((i-k)**2 + (j-k)**2)/(2*sigma**2) )
    ### END YOUR CODE

    return kernel

def partial_x(image):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None
    K = np.array([[1/2,0,-1/2],])
    ### YOUR CODE HERE
    out = conv(image, K)
    ### END YOUR CODE

    return out

def partial_y(image):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None
    K = np.array([
        [1/2],
        [0],
        [-1/2]])
    ### YOUR CODE HERE
    out = conv(image,K)
    ### END YOUR CODE

    return out

def gradient(image):
    """ Returns gradient magnitude and direction of input img.

    Args:
        image: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(image.shape)
    theta = np.zeros(image.shape)
    Gx = partial_x(image)
    Gy = partial_y(image)
    ### YOUR CODE HERE
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    theta = np.arctan2(Gy, Gx)
    theta = (np.rad2deg(theta) + 180) % 360
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45


    ### BEGIN YOUR CODE
    for i in range(1, H-1):
        for j in range(1, W-1):
            ang = int(theta[i][j]%360)
            if (ang%180 == 0):
                l = [G[i][j-1], G[i][j+1]]
            elif (ang%180 == 45):
                l = [G[i-1][j-1], G[i+1][j+1]]
            elif (ang%180 == 90):
                l = [G[i-1][j], G[i+1][j]]
            elif (ang%180 == 135):
                l = [G[i-1][j+1], G[i+1][j-1]]
            if G[i,j] >= np.max(l):
                out[i,j] = G[i,j]
            else:
                out[i, j] = 0
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array which represents strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img > high
    weak_edges = (img < high) & (img > low)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    Queue = []
    for i in range(len(indices)):
        Queue.append((indices[i, 0], indices[i, 1]))
    while( len(Queue) !=0 ):
        (y, x) = Queue.pop(0)
        neighbor = get_neighbors(y, x, H, W)
        for j in range(len(neighbor)):
            if weak_edges[neighbor[j][0], neighbor[j][1]] == True:
                edges[neighbor[j][0], neighbor[j][1]] = True
                Queue.append((neighbor[j][0], neighbor[j][1]))
                weak_edges[neighbor[j][0], neighbor[j][1]] = False
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # Smoothing by gaussian kernel
    kernel = gaussian_kernel(5, sigma)
    img = conv(img, kernel)
    
    # Get Gradient and direction of img
    G, theta = gradient(img)
    
    # Non maximum suppression
    img = non_maximum_suppression(G, theta)
    
    # Double thresholding
    strong_edge, weak_edge = double_thresholding(img, high, low)
    
    # Edge tracking
    edge = link_edges(strong_edge, weak_edge)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #there is a bug with just letting diag_len * 2.0 + 1 as argument
    # you need to explicitly cast it since the 2.0 makes it a float
    rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2.0 + 1))
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(len(ys)):
        x, y = xs[i], ys[i]
        rho_t = x*cos_t + y*sin_t
        acc_rho = np.int64(np.floor(rho_t)) + diag_len
        acc_theta = np.arange(0, 180)
        accumulator[acc_rho, acc_theta] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas
