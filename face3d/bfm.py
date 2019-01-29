"""
3D morphable model management.
This code is based on Yao Feng's code (https://github.com/YadiraF/face3d).
"""

import numpy as np
import scipy.io as sio
import face_alignment as fa

class MorphabelModel(object):
    """
    nver: number of vertices. 
    ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
    shapeMU: [3*nver, 1]. *
    shapePC: [3*nver, n_shape_para]. *
    shapeEV: [n_shape_para, 1]. ~
    expMU: [3*nver, 1]. ~ 
    expPC: [3*nver, n_exp_para]. ~
    expEV: [n_exp_para, 1]. ~
    texMU: [3*nver, 1]. ~
    texPC: [3*nver, n_tex_para]. ~
    texEV: [n_tex_para, 1]. ~
    tri: [ntri, 3] (start from 1, should sub 1 in python and c++). *
    tri_mouth: [114, 3] (start from 1, as a supplement to mouth triangles). ~
    kpt_ind: [68,] (start from 1). ~
    """
    def __init__(self, model_path, model_type='BFM'):
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.model = load(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]        
        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))
        
    def get_expression(self, type='random'):  # @ReservedAssignment
        if type == 'zero':
            expression = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            expression = -1.5 + 3 * np.random.random([self.n_exp_para, 1])
            expression[6:, 0] = 0
        return expression    
        
    def get_shape(self, type='random'):  # @ReservedAssignment
        if type == 'zero':
            shape = np.zeros((self.n_shape_para, 1))
        elif type == 'random':
            shape = np.random.rand(self.n_shape_para, 1) * 1e04
        return shape
    
    def get_texture(self, type='random'):  # @ReservedAssignment
        if type == 'zero':
            texture = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            texture = np.random.rand(self.n_tex_para, 1)
        return texture
    
    def fit(self, keypoints, keypoints_index, max_iter=4):   
        fitted_shape, fitted_expression, scale, rotation, translation = fit_points(keypoints, keypoints_index, self.model, n_shape=self.n_shape_para, n_expression=self.n_exp_para, max_iter=max_iter)
        angles = fa.rotation_matrix_to_euler_angles_2(rotation)
        return fitted_shape, fitted_expression, scale, angles, translation
    
    def generate_colors(self, texture):
        colors = self.model['texMU'] + self.model['texPC'].dot(texture * self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors) / 3)], 'F').T / 255.  
        return colors
    
    def generate_vertices(self, shape, expression):
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape) + self.model['expPC'].dot(expression)
        vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T
        return vertices

def estimate_expression(keypoints, shapeMU, expPC, expEV, shape, scale, rotation, translation, lamb=2000):
    keypoints = keypoints.copy()
    
    assert(shapeMU.shape[0] == expPC.shape[0])
    assert(shapeMU.shape[0] == keypoints.shape[1] * 3)

    dof = expPC.shape[1]

    n_keypoints = keypoints.shape[1]
    sigma = expEV
    translation = np.array(translation)
    p = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    a = scale * p.dot(rotation)

    pc_3d = np.resize(expPC.T, [dof, n_keypoints, 3]) 
    pc_3d = np.reshape(pc_3d, [dof * n_keypoints, 3]) 
    pc_2d = pc_3d.dot(a.T) 
    pc = np.reshape(pc_2d, [dof, -1]).T

    mu_3d = np.resize(shapeMU, [n_keypoints, 3]).T

    shape_3d = shape

    b = a.dot(mu_3d + shape_3d) + np.tile(translation[:, np.newaxis], [1, n_keypoints])
    b = np.reshape(b.T, [-1, 1])

    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma**2)
    keypoints = np.reshape(keypoints.T, [-1, 1])
    equation_right = np.dot(pc.T, keypoints - b)

    exp_para = np.dot(np.linalg.inv(equation_left), equation_right)
    
    return exp_para

def estimate_shape(keypoints, shapeMU, shapePC, shapeEV, expression, scale, rotation, translation, lamb = 3000):
    keypoints = keypoints.copy()
    
    assert(shapeMU.shape[0] == shapePC.shape[0])
    assert(shapeMU.shape[0] == keypoints.shape[1] * 3)

    dof = shapePC.shape[1]

    n_keypoints = keypoints.shape[1]
    sigma = shapeEV
    translation = np.array(translation)
    p = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    a = scale * p.dot(rotation)

    pc_3d = np.resize(shapePC.T, [dof, n_keypoints, 3])
    pc_3d = np.reshape(pc_3d, [dof*n_keypoints, 3]) 
    pc_2d = pc_3d.dot(a.T.copy())
    
    pc = np.reshape(pc_2d, [dof, -1]).T

    mu_3d = np.resize(shapeMU, [n_keypoints, 3]).T

    exp_3d = expression

    b = a.dot(mu_3d + exp_3d) + np.tile(translation[:, np.newaxis], [1, n_keypoints])
    b = np.reshape(b.T, [-1, 1])

    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma**2)
    keypoints = np.reshape(keypoints.T, [-1, 1])
    equation_right = np.dot(pc.T, keypoints - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para

def fit_points(keypoints, keypoints_index, model, n_shape, n_expression, max_iter = 4):
    keypoints = keypoints.copy().T

    sp = np.zeros((n_shape, 1), dtype = np.float32)
    ep = np.zeros((n_expression, 1), dtype = np.float32)

    X_ind_all = np.tile(keypoints_index[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_shape]
    expPC = model['expPC'][valid_ind, :n_expression]

    for _ in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T

        cam_matrix = fa.estimate_affine_matrix(X.T, keypoints.T)
        scale, rotation, translation = fa.decompose_camera_matrix(cam_matrix)

        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        ep = estimate_expression(keypoints, shapeMU, expPC, model['expEV'][:n_expression,:], shape, scale, rotation, translation[:2], lamb=20)

        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        shape = estimate_shape(keypoints, shapeMU, shapePC, model['shapeEV'][:n_shape,:], expression, scale, rotation, translation[:2], lamb=40)

    return shape, ep, scale, rotation, translation
       
def load(model_path):
    c = sio.loadmat(model_path)
    
    model = c['model']
    model = model[0,0]
    
    model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expEV'] = model['expEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)
    model['tri'] = model['tri'].T.copy(order = 'C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order = 'C').astype(np.int32) - 1
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model