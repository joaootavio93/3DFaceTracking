""" 
Contains common functions.
"""

import cv2

def create_alpha_blending(fg_img, bg_img):
    """ Blends two images using the alpha channel.
    Parameters:
        fg_img (mat): The foreground image.
        bg_img (mat): The background image.
    Returns:
        mat: The blended image.
    """
    b, g, r, a = cv2.split(fg_img)
    foreground = cv2.merge((b, g, r))
    alpha = cv2.merge((a, a, a))
    foreground = foreground.astype(float)
    background = bg_img.astype(float)
    alpha = alpha.astype(float)
    blended_img = cv2.add(foreground, background)
    return blended_img

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
