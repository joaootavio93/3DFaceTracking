"""
Contains methods for 3DMM face fitting purposes.
This code is based on Yao Feng's code (https://github.com/YadiraF/face3d).
"""

import cv2
import inspect
import os
import utils
import numpy as np
import face_alignment as fa
from face3d import bfm

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

morphable_model = bfm.MorphabelModel(current_dir + '/face3d/BFM/BFM.mat') # The default BFM (Base Face Model).
shape = morphable_model.get_shape('random')
expression = morphable_model.get_expression('random')
texture = morphable_model.get_texture('random')
vertices = morphable_model.generate_vertices(shape, expression)
colors = morphable_model.generate_colors(texture)
colors = np.minimum(np.maximum(colors, 0), 1)

def fit_3dmm(rotation, translation, scale, width, height, max_iter=3):
    """ Fits a 3D face morphable model on a 2D face.
    
    Parameters:
        rotation (ndarray): The 2D face rotation matrix.
        translation (list): The 2D face translation vector.
        scale (float): The 2D face scale.
        width (int): The source image width.
        height (int): The source image height.
        max_iter (int): The number of fits iterates.
        
    Returns:
        mat: The fitted 3DMM image with alpha channel.
        ndarray: The fitted 3DMM 68 keypoints.
    """    
    transform_vertices = fa.similarity_transform(vertices, rotation, translation, scale)

    keypoints = transform_vertices[morphable_model.kpt_ind, : 2]
    keypoints_index = morphable_model.kpt_ind
      
    fit_shape, fit_exp, fit_scale, fit_rotation, fit_t3d = morphable_model.fit(keypoints, keypoints_index, max_iter=max_iter)
      
    fitted_vertices = morphable_model.generate_vertices(fit_shape, fit_exp)
    fit_rotation = fa.euler_angles_to_rotation_matrix(fit_rotation)
    
    transform_vertices = fa.similarity_transform(fitted_vertices, fit_rotation, [0, 0, 0], fit_scale)     
    img_2d_vertices = fa.to_coord_system(transform_vertices, width, height)    
    
    fit_kpts = img_2d_vertices[morphable_model.kpt_ind, : 2]

    fit_img = fa.render_colors(img_2d_vertices, morphable_model.triangles, colors, width, height)
     
    fit_img = fit_img * 255
    fit_img = fit_img.astype('uint8')
    fit_img = cv2.cvtColor(fit_img, cv2.COLOR_RGB2BGR)
    
    fit_img = utils.add_alpha_channel(fit_img)

    return fit_img, fit_kpts
