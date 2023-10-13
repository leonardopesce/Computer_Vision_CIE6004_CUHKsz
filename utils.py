import os
import cv2
import numpy as np
from PIL import Image

import scipy.signal
import scipy.sparse
import scipy.linalg
from skimage.segmentation import find_boundaries

def setup_variables(mask):
    height, width = mask.shape[:2]

    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY) # fix here
    inner_mask, boundary_mask = process_mask(mask)

    pixel_ids, inner_ids, boundary_ids, mask_ids = get_ids(mask, inner_mask, boundary_mask)

    inner_pos, boundary_pos, mask_pos = get_pos(mask_ids, inner_ids, boundary_ids, pixel_ids)

    return height, width, mask, inner_mask, boundary_mask, pixel_ids, inner_ids, boundary_ids, mask_ids, inner_pos, boundary_pos, mask_pos

def open_and_print_image(file_path, is_mask = False):
    img = Image.open(check_if_jpg_or_png(file_path))
    if is_mask:
        img = img.convert("L")
    display(img)
    img = np.array(img).astype(np.float64) / 255
    return img

def check_if_jpg_or_png(file_path):
    if not os.path.isfile(file_path + ".jpg"):
        return file_path + ".png"
    return file_path + ".jpg"

def save_and_print(img, folder_path, file_name):
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(folder_path, file_name))
    display(img)

def rgb2gray(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def process_mask(mask):
    boundary = find_boundaries(mask, mode="inner").astype(int)
    inner = mask - boundary
    return inner, boundary

def compute_laplacian(img):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    laplacian = scipy.signal.fftconvolve(img, kernel, mode="same")
    return laplacian

def compute_gradient(img, forward=True):
    if forward:
        kx = np.array([
            [0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 1, 0]
        ])
    else:
        kx = np.array([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    Gx = scipy.signal.fftconvolve(img, kx, mode="same")
    Gy = scipy.signal.fftconvolve(img, ky, mode="same")
    return Gx, Gy

def get_pixel_ids(img):
    pixel_ids = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1])
    return pixel_ids

def get_masked_values(values, mask):
    assert values.shape[:2] == mask.shape
    if len(values.shape) == 3:
        nonzero_idx = np.array([np.nonzero((mask)) for i in range(values.shape[2])]) # get mask 1
    else:
        nonzero_idx = np.nonzero(mask) # get mask 1
    return values[nonzero_idx]

def get_alpha_blended_img(src, target, alpha_mask):
    return src * alpha_mask + target * (1 - alpha_mask)

def dilate_img(img, k):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def create_matrix_A(mask_ids, inner_ids, boundary_ids, inner_pos, boundary_pos, width):
    A = scipy.sparse.lil_matrix((len(mask_ids), len(mask_ids)))

    n1_pos = np.searchsorted(mask_ids, inner_ids - 1)
    n2_pos = np.searchsorted(mask_ids, inner_ids + 1)
    n3_pos = np.searchsorted(mask_ids, inner_ids - width)
    n4_pos = np.searchsorted(mask_ids, inner_ids + width)

    A[inner_pos, n1_pos] = A[inner_pos, n2_pos] = A[inner_pos, n3_pos] = A[inner_pos, n4_pos] = 1
    A[inner_pos, inner_pos] = -4 

    A[boundary_pos, boundary_pos] = 1
    return A.tocsr()

def get_ids(mask, inner_mask, boundary_mask):
    # create unique id for each pixel
    pixel_ids = get_pixel_ids(mask) 
    # create unique id for each pixel in the inner region 
    inner_ids = get_masked_values(pixel_ids, inner_mask).flatten()
    # create unique id for each pixel in the boundary region
    boundary_ids = get_masked_values(pixel_ids, boundary_mask).flatten()
    # create unique id for each pixel in the boundary + inner region
    mask_ids = get_masked_values(pixel_ids, mask).flatten() # boundary + inner
    return pixel_ids, inner_ids, boundary_ids, mask_ids

def get_pos(mask_ids, inner_ids, boundary_ids, pixel_ids):
    inner_pos = np.searchsorted(mask_ids, inner_ids) 
    boundary_pos = np.searchsorted(mask_ids, boundary_ids)
    mask_pos = np.searchsorted(pixel_ids.flatten(), mask_ids)

    return inner_pos, boundary_pos, mask_pos

def construct_b(mask_ids, inner_pos, boundary_pos, inner_gradient_values, boundary_pixel_values):
    b = np.zeros(len(mask_ids))
    b[inner_pos] = inner_gradient_values
    b[boundary_pos] = boundary_pixel_values
    return b