""" 
Contains common functions.
"""
import cv2
import numpy as np

def add_alpha_channel(img):
    """ Adds alpha channel to an image.
    
    Parameters:
        img (mat): The image which will be received the alpha channel.
        
    Returns:
        mat: The image with alpha channel.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          
    _, alpha = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY)
    
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    alpha_img = cv2.merge(rgba, 4)
    
    return alpha_img

def create_transparent_overlay(foreground, background, pos=(0,0)):
    """ Overlays a foreground image on a background image using the alpha channel.
    
    Parameters:
        background (mat): The background image.
        foreground (mat): The foreground image.
        pos (list): The overlay position.
        
    Returns:
        mat: The new blended image.
    """
    h, w, _ = foreground.shape
    rows, cols, _ = background.shape
    y, x = pos[0], pos[1]
    
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(foreground[i][j][3] / 255.0)
            background[x + i][y + j] = alpha * foreground[i][j][:3] + (1 - alpha) * background[x + i][y + j]
    return background

def create_transparent_overlay_faster(foreground, background, width, height):
    result = np.zeros((height, width, 3), np.uint8)
    alpha = foreground[:, :, 3] / 255.0
    result[:, :, 0] = (1. - alpha) * background[:, :, 0] + alpha * foreground[:, :, 0]
    result[:, :, 1] = (1. - alpha) * background[:, :, 1] + alpha * foreground[:, :, 1]
    result[:, :, 2] = (1. - alpha) * background[:, :, 2] + alpha * foreground[:, :, 2]
    return result

def landmarks_to_np_array(fore_lmks, back_lmks):
    """ Converts the 3DDFA and face3d face landmarks to numpy array.
    
    Parameters:
        fore_lmks (list): The landmarks points from 3DDFA.
        back_lmks (list): The landmarks points from face3d.
        
    Returns:
        ndarray: The landmarks points from 3DDFA as numpy array.
        ndarray: The landmarks points from face3d as numpy array.
    """
    np_fore_lmks = []
    np_back_lmks = []
    
    lmks_x, lmks_y = back_lmks[0].T, back_lmks[1].T
    
    i = 0
    for x, y in fore_lmks:
        np_fore_lmks.append([x, y])
        np_back_lmks.append([lmks_x[i], lmks_y[i]])
        i += 1

    return np.asarray(np_fore_lmks), np.asarray(np_back_lmks)
    
def resize_image(img, width, height):
    """ Resizes an image.
    
    Parameters:
        img (mat): The image to be resized.
        width (int): The resize width value.
        height (int): THe resize height value.
        
    Returns:
        mat: The resized image.
    """
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
