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
    new_gray = gray - black_level
    balance = (I - [[black_level]]) * [[new_gray[1] / new_gray]]
    return (balance / (2**16 - 1)).astype(np.float32) # Replace this with your implementation


# ======================= color_transform =======================
# Input:
#   I: an RGB image -- a numpy array of shape (height, width, 3)
#   M: a 3x3 matrix, to be multiplied with each RGB triple in I
# Output:
#   The image with each RGB triple multiplied by M
def color_transform(I, M):
    
    # A3TODO: Complete this function
    transformed = np.tensordot(I, M, axes=(2, 1))
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
    length = img.shape[1]
    height = img.shape[0]
    
    for i in range(0, height):
        for j in range(0, length):
            if (j+k < length):
                new_img[i][j] = img[i][j+k]
            else:
                new_img[i][j] = img[i][length-1]

    return new_img


# ======================= rotate_image =======================
# Input:
#   img: 2D numpy array of a grayscale image
#   k: The angle (in degrees) to be rotated counter-clockwise around the image center
#   interp_mode: 0 for nearest neighbor, 1 for bilinear
# Output:
#   A 2D array of img rotated around the original image's center by k degrees
def rotate_image(img, k, interp_mode):
    new_img = np.zeros(img.shape, np.uint8)
    # A3TODO: Complete this function
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    length = img.shape[1]
    height = img.shape[0]
    rad = np.radians(k)
        
    if (interp_mode == 0): 
        #nearest neighbor
        for i in range(0, height):
            for j in range(0, length):
                row = round(((j - center_x) * np.cos(rad)) - ((i - center_y) * np.sin(rad)) + center_x)
                col = round(((j - center_x) * np.sin(rad)) + ((i - center_y) * np.cos(rad)) + center_y)
                if (row < length and row >= 0 and col < height and col >= 0):
                    new_img[i][j] = img[col][row]


        return new_img 

    else: 
        #bilinear interpolation
        for i in range(0, height):
            for j in range(0, length):
                row = ((j - center_x) * np.cos(rad)) - ((i - center_y) * np.sin(rad)) + center_x
                col = ((j - center_x) * np.sin(rad)) + ((i - center_y) * np.cos(rad)) + center_y
                a = col - np.floor(col)
                b = row - np.floor(row)
                col0 = np.floor(col).astype(np.int)
                row0 = np.floor(row).astype(np.int)
                col1 = np.ceil(col).astype(np.int)
                row1 = np.ceil(row).astype(np.int)
                #q00
                if (col0 >= 0 and col0 <= height - 1 and row0 >= 0 and row0 <= length - 1):
                    q00 = img[col0][row0]
                else:
                    q00 = 0
                
                #q01
                if(col1 >= 0 and col1 <= height - 1 and row0 >= 0 and row0 <= length - 1):
                    q01 = img[col1][row0]
                else:
                    q01 = 0
                    
                #q10
                if(col0 >= 0 and col0 <= height - 1 and row1 >= 0 and row1 <= length - 1):
                    q10 = img[col0][row1]
                else:
                    q10 = 0
                    
                #q11
                if(col1 >= 0 and col1 <= height - 1 and row1 >= 0 and row1 <= length - 1):
                    q11 = img[col1][row1]
                else:
                    q11 = 0
                    
                new_img[i][j] = ((1 - a) * (1 - b) * q00) + ((1 - a) * b * q10) + (a * (1 - b) * q01) + (a * b * q11)

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
def undistort_image(img, k1, k2, M, interp_mode):
    
    Mi = np.linalg.inv(M)
    output = np.zeros_like(img)
    # A3TODO: Complete this function
    length = output.shape[1]
    height = output.shape[0]
    for i in range(0, height):
        for j in range(0, length):
            dist_coord = Mi @ np.array([j, i, 1])
            dist_i = dist_coord[1]
            dist_j = dist_coord[0]
            r = np.sqrt(dist_i ** 2 + dist_j ** 2)
            sr = 1 + k1 * (r ** 2) + k2 * (r ** 4)
            dist_x_coord = sr * dist_j
            dist_y_coord = sr * dist_i
            new_img_xy = M @ np.array([dist_x_coord, dist_y_coord, 1])
            
            if(interp_mode == 0): 
                #nearest_neighbor
                new_img_x = round(new_img_xy[0])
                new_img_y = round(new_img_xy[1])
                if (new_img_x < length and new_img_x >= 0 and new_img_y < height and new_img_y >= 0):
                    output[i][j] = img[new_img_y][new_img_x]

            else:
                #binary interpolation version of undistort_image
                row = new_img_xy[0]
                col = new_img_xy[1]
                a = col - np.floor(col)
                b = row - np.floor(row)
                col0 = np.floor(col).astype(np.int)
                row0 = np.floor(row).astype(np.int)
                col1 = np.ceil(col).astype(np.int)
                row1 = np.ceil(row).astype(np.int)
                #q00
                if (col0 >= 0 and col0 <= height - 1 and row0 >= 0 and row0 <= length - 1):
                    q00 = img[col0][row0]
                else:
                    q00 = 0
                
                #q01
                if(col1 >= 0 and col1 <= height - 1 and row0 >= 0 and row0 <= length - 1):
                    q01 = img[col1][row0]
                else:
                    q01 = 0
                    
                #q10
                if(col0 >= 0 and col0 <= height - 1 and row1 >= 0 and row1 <= length - 1):
                    q10 = img[col0][row1]
                else:
                    q10 = 0
                    
                #q11
                if(col1 >= 0 and col1 <= height - 1 and row1 >= 0 and row1 <= length - 1):
                    q11 = img[col1][row1]
                else:
                    q11 = 0
                    
                output[i][j] = ((a - 1) * (b - 1) * q00) + ((1 - a) * b * q10) + (a * (1 - b) * q01) + (a * b * q11)
        
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
    # A3TODO: Complete this function
    center = dim // 2
    
    f = np.zeros([dim, dim])
    for i in range(0, dim):
        for j in range(0, dim):
            f[i,j] = np.exp(-((i-center)**2 + (j-center)**2)/(2* sigma**2)) / (2 * np.pi * sigma**2)
    
    return f


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
    output = np.zeros_like(I).astype(np.float64)
    image_height = I.shape[0]
    image_length = I.shape[1]
    f_height = f.shape[0]
    f_length = f.shape[1]
    
    for p in range(0, image_length):
        for q in range(0, image_height):
            
                s = 0.0
                for i in range(0, f_height):
                    for j in range(0, f_length):
                        if ((q - f_height // 2 + i) < image_height and (q - f_height // 2 + i) >= 0 
                            and (p - f_length // 2 + j) >= 0 and (p - f_length // 2 + j) < image_length):
                            s = s + f[i][j] * I[q - f_height // 2 + i][p - f_length // 2 + j]
                output[q][p] = s
            
    return np.clip(output, 0, 255).astype(np.uint8)


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
    image_height = I.shape[0]
    image_length = I.shape[1]
    f_height = f.shape[0]
    f_length = f.shape[1]
    
    ctr_row = f.shape[0] // 2
    ctr_col = f.shape[1] // 2
    
    f_along_row = f[:][ctr_row-1] / np.sum(f[:][ctr_row-1])
    f_along_col = f[ctr_col-1][:] / np.sum(f[ctr_col-1][:])
    
    temp = np.zeros_like(I)
    for p in range(0, image_length):
        for q in range(0, image_height):
            
            s = 0.0
            for i in range(0, f_height):
                if ((q - f_height // 2 + i) < image_height and (q - f_height // 2 + i) >= 0):
                    s = s + f_along_row[i] * I[q - f_height // 2 + i][p]
            temp[q][p] = np.clip(s, 0, 255).astype(np.uint8)
    
    for x in range(0, image_length):
        for y in range(0, image_height):
            
            t = 0.0
            for j in range(0, f_length):
                if ((x - f_length // 2 + j) >= 0 and (x - f_length // 2 + j) < image_length):
                    t = t + f_along_col[j] * temp[y][x - f_length // 2 + j]
            output[y][x] = np.clip(t, 0, 255).astype(np.uint8)
    
    return np.clip(output, 0, 255).astype(np.uint8)


# ======================= unsharp_mask =======================
# This function essentially subtracts a (scaled) blurred version of an image from (scaled version of) itself
# Input:
#   I: A 2D numpy array containing pixels of an image
#   sigma: Gassian std.dev. for blurring
#   w: Sharpening weight
# Output:
#   A sharpened version of I
def unsharp_mask(I, sigma, w):
    # A3TODO: Complete this function
    output = np.zeros_like(I)
    r = np.ceil(3 * sigma).astype(np.int)
    gaussian_blur = gen_gaussian_filter(2*r, sigma)
    blurred = convolve_sep(I, gaussian_blur)
    output = np.clip((1 + w) * I - w * blurred, 0, 255)

    return output.astype(np.uint8)
