#!/usr/bin/env python
import math
import numpy as np
from PIL import Image
import tifffile

# *************************************************************
# *                    From Photography Notebook              *
# *************************************************************

# ======================= white_balance =======================
# Input:
#   I: an RGB image -- a numpy array of shape (height, width, 3)
#   black_level: an RGB offset to be subtracted from all the pixels
#   gray: the RGB color of a gray object (includes the black level)
# Output:
#   The corrected image: black level subtracted, then color channels scale to make gray come out gray
def white_balance(I, black_level, gray):

    # A3TODO: Complete this function
    return I # Replace this with your implementation


# ======================= color_transform =======================
# Input:
#   I: an RGB image -- a numpy array of shape (height, width, 3)
#   M: a 3x3 matrix, to be multiplied with each RGB triple in I
# Output:
#   The image with each RGB triple multiplied by M
def color_transform(I, M):
    
    # A3TODO: Complete this function
    return I # Replace this with your implementation


# *************************************************************
# *                    From Distortion Notebook               *
# *************************************************************

# ======================= shift_image_to_left =======================
# Input:
#   img: 2D numpy array of a grayscale image
#   k: The number of units/pixels to be shifted to the left (you can assume k < width of image)
# Output:
#   A 2D array of img shifted to the left by k pixels
#  For points that fall out of range on the right side, repeat the rightmost pixel. 
def shift_image_to_left(img, k):
    new_img = np.zeros(img.shape, np.uint8)
    # A3TODO: Complete this function

    return new_img


# ======================= rotate_image =======================
# Input:
#   img: 2D numpy array of a grayscale image
#   k: The angle (in degrees) to be rotated counter-clockwise around the image center
#   interp_mode: 0 for nearest neighbor, 1 for bilinear
# Output:
#   A 2D array of img rotated around the original image's center by k degrees
def rotate_image(img, k, interp_mode=0):
    new_img = np.zeros(img.shape, np.uint8)
    # A3TODO: Complete this function
            
    if interp_mode == 0:
        # nearest neighbor


    else:
        # bilinear



    return new_img 


# ======================= undistort_image =======================
# Input:
#   img: A distorted image, with coordinates in the distorted space
#   k1, k2: distortion model coefficients (see explanation above)
#   M: affine transformation from pixel coordinates to distortion-model coordinates
#   interp_mode: 0 for nearest neighbor, 1 for bilinear
# Output:
#   An undistorted image, with pixels in the image coordinates
# Write down the formula for calculating the distortion model first (see exercise above)
# Put black in for points that fall out of bounds
def undistort_image(img, k1, k2, M, interp_mode=0):
    Mi = np.linalg.inv(M)
    output = np.zeros_like(img)
    # A3TODO: Complete this function
    h, w = img.shape[:2]
    
    if interp_mode == 0:
        # nearest neighbor

        
    else:
        # bilinear


    return output


# *************************************************************
# *                    From Convolution Notebook              *
# *************************************************************

# ======================= gen_gaussian_filter =======================
# Input:
#   dim: size of the filter in both x and y direction
#   sigma: standard deviation of the gaussian filter
# Output:
#   A 2-dimensional numpy array of size dim*dim
#   (Note that the array should be normalized)
# Hint: Use linspace or mgrid from numpy
def gen_gaussian_filter(dim, sigma):
    # A3 implement
    pass # Replace this line with your implementation


# ======================= convolve =======================
# Input:
#   I: A 2D numpy array containing pixels of an image
#   f: A squared/non-squared filter of odd/even-numbered dimensions
# Output:
#   A 2D numpy array resulting from applying the convolution filter f to I
#   All the entries of the array should be of type uint8, and restricted to [0,255]
#   You may use clip and astype in numpy to enforce this
# Note: When convolving, do not operate on the entries outside of the image bound,
#           i.e. clamp the ranges to the width and height of the image
#       Tie-breaking: If f has an even number of dimensions in some direction (assume the dimension is 2r),
#           sweep through [i-r+1, i+r] (i.e. length of left half = length of right half - 1)
#           With odd # of dimensions (2r+1), you would sweep through [i-r, i+r].
def convolve(I, f):
    output = np.zeros_like(I)
    
    # A3TODO: Complete this function

    return output


# ======================= convolve_sep =======================
# Input:
#   I: A 2D numpy array containing pixels of an image
#   f: A squared/non-squared filter of odd/even-numbered dimensions
# Output:
#   A 2D numpy array resulting from applying the convolution filter f to I
#   All the entries of the array should be of type uint8, and restricted to [0,255]
#   You may use clip and astype in numpy to enforce this
# Note: When convolving, do not operate on the entries outside of the image bound,
#           i.e. clamp the ranges to the width and height of the image in the for loop
#       Tie-breaking: If f has an even number of dimensions in some direction (assume the dimension is 2r),
#           sweep through [i-r+1, i+r] (i.e. length of left half = length of right half - 1)
#           With odd # of dimensions (2r+1), you would sweep through [i-r, i+r].
#       You will convolve with respect to the direction corresponding to I.shape[0] first, then I.shape[1]
def convolve_sep(I, f):
    output = np.zeros_like(I)
        
    # A3TODO: Complete this function

    return output


# ======================= unsharp_mask =======================
# This function essentially subtracts a (scaled) blurred version of an image from (scaled version of) itself
# Input:
#   I: A 2D numpy array containing pixels of an image
#   sigma: Gassian std.dev. for blurring
#   w: Sharpening weight
# Output:
#   A sharpened version of I
def unsharp_mask(I, sigma, w):
    output = np.zeros_like(I)
    # A3TODO: Complete this function

    return output
