import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
from skimage.segmentation import find_boundaries


def convert_image_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def triple_image(a, b, c):
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(131)
    plt.imshow(convert_image_color(a))
    plt.subplot(132)
    plt.imshow(convert_image_color(b))
    plt.subplot(133)
    plt.imshow(convert_image_color(c))


def double_image(a, b):
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(convert_image_color(a))
    plt.subplot(132)
    plt.imshow(convert_image_color(b))

def double_image_gray(a, b):
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(a)
    plt.subplot(132)
    plt.imshow(b)

def single_image(a):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(convert_image_color(a))


def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A

def laplacian_matrix(x, y):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A"""
    # Calculate the total number of grid points
    N = x * y

    # Create a sparse matrix in the Compressed Sparse Row (CSR) format
    A = scipy.sparse.lil_matrix((N, N))

    # Fill the Laplacian matrix with appropriate values
    for i in range(N):
        # Diagonal element
        A[i, i] = 4.0

        # Off-diagonal elements (neighbors)
        if i % x != 0:         # Left neighbor
            A[i, i - 1] = -1.0
        if (i + 1) % x != 0:   # Right neighbor
            A[i, i + 1] = -1.0
        if i >= x:             # Top neighbor
            A[i, i - x] = -1.0
        if i < N - x:          # Bottom neighbor
            A[i, i + x] = -1.0

    # Convert the lil_matrix to CSR format for better performance
    return A.tocsr()


def finite_forward_gradient_xy(mat):
    kernel_x = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    kernel_y = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    grad_x = scipy.signal.fftconvolve(mat, kernel_x, mode="same")
    grad_y = scipy.signal.fftconvolve(mat, kernel_y, mode="same")
    return grad_x, grad_y


def finite_backward_gradient_xy(mat):
    kernel_x = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    kernel_y = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    grad_x = scipy.signal.fftconvolve(mat, kernel_x, mode="same")
    grad_y = scipy.signal.fftconvolve(mat, kernel_y, mode="same")
    return grad_x, grad_y

def finite_full_gradient_xy(mat):
    kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 1, 0], [0, -4, 0], [0, 1, 0]],
                       [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    grad = scipy.signal.fftconvolve(mat, kernel, mode="same")
    return grad

def finite_full_gradient_xy_single(mat):
    kernel = np.array([[0, 1, 0], 
                       [1, -4, 1], 
                       [0, 1, 0]])
    grad = scipy.signal.fftconvolve(mat, kernel, mode="same")
    return grad


def finite_partial_gradient_xy(mat):
    kernel_x = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    kernel_y = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    grad_x = scipy.signal.fftconvolve(mat, kernel_x, mode="same")
    grad_y = scipy.signal.fftconvolve(mat, kernel_y, mode="same")
    return grad_x, grad_y


def split_function_per_color(source, func, channels):
    if channels == 1:
        return func(source)
    for i in range(channels):
        source[:,:,i] = func(source[:,:,i])
    return source
    

def process_mask(mask):
    boundary = find_boundaries(mask, mode="inner").astype(int)
    inner = mask - boundary
    return inner, boundary