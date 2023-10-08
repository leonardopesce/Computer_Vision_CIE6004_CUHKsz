import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve


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