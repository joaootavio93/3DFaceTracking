"""
Contains functions for face aligment purposes.
This code is based on Jianzhu Guo's code (https://github.com/cleardusk/3DDFA) and Yao Feng's code (https://github.com/YadiraF/face3d).
"""
import inspect
import math
import os
import pickle
import sys
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from ddfa import mobilenet_v1
from face3d.mesh import mesh_core_cython as mcc

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_dir + '/ddfa/')

arch = 'mobilenet_1'
checkpoint_fp = 'ddfa/models/phase1_wpdc_vdc_v2.pth.tar' # The 3DDFA pose classifier model
mode = 'gpu' # Set 'cpu' for running in CPU

std_size = 120 # The 3DDFA network input size.

checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']  # @UnusedVariable
model = getattr(mobilenet_v1, arch)(num_classes=62)
model_dict = model.state_dict()

for k in checkpoint.keys():
    model_dict[k.replace('module.', '')] = checkpoint[k]
    
model.load_state_dict(model_dict)
if mode == 'gpu':
    cudnn.benchmark = True
    model = model.cuda()
model.eval()

class NormalizeGjz(object):
    """ See https://github.com/cleardusk/3DDFA.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

class ToTensorGjz(object):
    """ See https://github.com/cleardusk/3DDFA.
    """
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))  # @UndefinedVariable
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
params_dir = current_dir + '/ddfa/train.configs/' # The directory containing the 3DDFA parameters

def get_suffix(filename):
    """ Extracts a file extension.
    
    Parameters:
        filename (str): The file name.
    
    Returns:
        str: The file extension. 
    """
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def load_param(path):
    """ Loads files containing npy or pkl extensions.
    
    Parameters:
        path (str): The file path.
    
    Returns:
        type: Either numpy or pickle file. 
    """
    suffix = get_suffix(path)
    if suffix == 'npy':
        return np.load(path)
    elif suffix == 'pkl':
        return pickle.load(open(path, 'rb'))

"""
Loading the 3DDFA parameters.
"""
keypoints = load_param(params_dir + 'keypoints_sim.npy')
w_shp = load_param(params_dir + 'w_shp_sim.npy')
w_exp = load_param(params_dir + 'w_exp_sim.npy')
meta = load_param(params_dir + 'param_whitening.pkl')
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = load_param(params_dir +  'u_shp.npy')
u_exp = load_param(params_dir +  'u_exp.npy')
u = u_shp + u_exp
w = np.concatenate((w_shp, w_exp), axis=1)
w_base = w[keypoints]
w_norm = np.linalg.norm(w, axis=0)
w_base_norm = np.linalg.norm(w_base, axis=0)
dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
paf = load_param(params_dir +  'Model_PAF.pkl')
u_filter = paf.get('mu_filter')
w_filter = paf.get('w_filter')
w_exp_filter = paf.get('w_exp_filter')
pncc_code = load_param(params_dir + 'pncc_code.npy')
"""
End of 3DDFA parameters loading.
"""

def to_coord_system(vertices, width, height):
    """ Changes vertices to image coordinate system (2D image)
    
    Parameters:
        vertices (ndarray): The 3D face vertices.
        width (int): The rendering width.
        height (int): The rendering height.
    
    Returns:
        ndarray: The new vertices in coordinate system.
    """
    proj_vertices = vertices.copy()

    proj_vertices[:, 0] = proj_vertices[:, 0] + width / 2
    proj_vertices[:, 1] = proj_vertices[:, 1] + height / 2
    proj_vertices[:, 1] = height - proj_vertices[:, 1] - 1
    
    return proj_vertices 

def decompose_camera_matrix(cam_matrix):
    """ Extracts the pose information from camera matrix.
    
    Parameters:
        cam_matrix (ndarray): The camera matrix.
        
    Returns:
        float: The camera matrix scale:
        ndarray: The extracted rotation matrix.
        ndarray: The extracted translation vector. 
    """
    translation = cam_matrix[:, 3]
    
    R1 = cam_matrix[0:1, :3]
    R2 = cam_matrix[1:2, :3]
    
    scale = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    rotation = np.concatenate((r1, r2, r3), 0)
    
    return scale, rotation, translation

def estimate_affine_matrix(x1, x2):
    """ See https://github.com/YadiraF/face3d.
    """
    x1 = x1.T; x2 = x2.T
    
    assert(x2.shape[1] == x1.shape[1])
    
    n = x2.shape[1]
    assert(n >= 4)

    mean = np.mean(x2, 1) 
    x2 = x2 - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x2**2, 0)))
    scale = np.sqrt(2) / average_norm
    x2 = scale * x2

    T = np.zeros((3, 3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean * scale
    T[2, 2] = 1

    mean = np.mean(x1, 1)
    x1 = x1 - np.tile(mean[:, np.newaxis], [1, n])

    average_norm = np.mean(np.sqrt(np.sum(x1**2, 0)))
    scale = np.sqrt(3) / average_norm
    x1 = scale * x1

    u = np.zeros((4,4), dtype = np.float32)
    u[0, 0] = u[1, 1] = u[2, 2] = scale
    u[:3, 3] = -mean * scale
    u[3, 3] = 1

    a = np.zeros((n * 2, 8), dtype = np.float32);
    X_homo = np.vstack((x1, np.ones((1, n)))).T
    a[:n, :4] = X_homo
    a[n:, 4:] = X_homo
    
    b = np.reshape(x2, [-1, 1])
 
    p_8 = np.linalg.pinv(a).dot(b)
    p = np.zeros((3, 4), dtype = np.float32)
    p[0, :] = p_8[:4, 0]
    p[1, :] = p_8[4:, 0]
    p[-1, -1] = 1

    p_affine = np.linalg.inv(T).dot(p.dot(u))
    
    return p_affine

def euler_angles_to_rotation_matrix(angles):
    """ Parses Euler angles to rotation matrix.
    
    Parameters:
        angles (array): The Euler angles containing pitch, yaw, roll.
        
    Returns:
        ndarray: The rotation matrix. 
    """
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    
    rx = np.array([[1, 0, 0], [0, math.cos(x),  -math.sin(x)], [0, math.sin(x),   math.cos(x)]])
    ry = np.array([[ math.cos(y), 0, math.sin(y)],[0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z),  math.cos(z), 0], [0, 0, 1]])
    
    rotation = rz.dot(ry.dot(rx))
    
    return rotation.astype(np.float32)

def euler_angles_to_rotation_matrix_2(pitch, yaw, roll):
    """ Parses Euler angles to rotation matrix.
    
    Parameters:
        pitch (float): The pitch value.
        yaw (float): The yaw value.
        roll (float): The roll value.
    Returns:
        ndarray: The rotation matrix. 
    """ 
    phi = pitch;
    gamma = yaw;
    theta = roll;
    
    r_x = np.array([ [1, 0, 0] , [0, np.cos(phi), np.sin(phi)] , [0, -np.sin(phi), np.cos(phi)] ]);
    r_y = np.array([ [np.cos(gamma), 0, -np.sin(gamma)] , [0, 1, 0] , [np.sin(gamma), 0, np.cos(gamma)] ]);
    r_z = np.array([ [np.cos(theta), np.sin(theta), 0] , [-np.sin(theta), np.cos(theta), 0] , [0, 0, 1] ]);
    
    rotation = np.matmul( r_x , np.matmul(r_y , r_z) );
    
    return rotation

def extract_pose_from_param(param):
    """ Extracts the pose information from 3DDFA params.
    
    Parameters:
        param (ndarray): Contains 3DMM params (12-pose, 40-shape, 10-expression).
        
    Returns:
        float: The pitch value.
        float: The yaw value.
        float: The roll value.
        float: The 3DMM scale value.
        ndarray: The rotation matrix.
        ndarray: The translation vector.
        ndarray: The camera matrix (rotation + translation).
    """
    param = param * param_std + param_mean
    pose = param[:12].reshape(3, -1)  
    scale, rotation, t3d = decompose_camera_matrix(pose)    
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation) 
    cam_matrix = np.concatenate((rotation, t3d.reshape(3, -1)), axis=1)
    return pitch, yaw, roll, scale, rotation, t3d, cam_matrix

def parse_param(param):
    """ Extract shape, expression and pose from 3DDFA params.
    
    Parameters:
        param (ndarray): Contains 3DMM params (12-pose, 40-shape, 10-expression). 
        
    Returns:
        ndarray: The pose param.
        ndarray:   
        ndarray: The shape param.
        ndarray: The expression param.
    """
    p_ = param[:12].reshape(3, -1)
    pose = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return pose, offset, alpha_shp, alpha_exp

def predict_landmarks(param, bbox, dense=False):
    """ Predicts the 68 face landmarks from a 3D face, scaling them by the face bounding box.
    
    Parameters:
        param (ndarray): Contains 3DMM params (12-pose, 40-shape, 10-expression). 
        bbox (array): The face bounding box.
        dense (bool): Predict dense vertices or not.
        
    Returns:
        ndarray: The 2D face landmarks.
        float: The landmarks scale factor calculated using the face bounding box.
    """
    landmarks = reconstruct_vertex(param, dense=dense)
    x, y, w, h = bbox
    scale_x = (w - x) / 120
    scale_y = (h - y) / 120
    landmarks[0, :] = landmarks[0, :] * scale_x + x
    landmarks[1, :] = landmarks[1, :] * scale_y + y
    scale = (scale_x + scale_y) / 2
    landmarks[2, :] *= scale
    return landmarks, scale

def predict_pose(img, bbox, dense=False):
    """ Calculates the face pose using the 3DDFA algorithm (https://github.com/cleardusk/3DDFA).
    Parameters:
        img (mat): The input image for the network (cropped and resized to 120).
        bbox (array): The source image face bounding box.
        dense (bool): Predict dense vertices or not.
        
    Returns:
        float: The pitch value.
        float: The yaw value.
        float: The roll value.
        float: The face vertex scale.
        ndarray: The rotation matrix.
        ndarray: The translation vector.
        ndarray: The pose matrix (rotation + translation).
        ndarray: The original face landmarks.
        float: The scale factor for calculating the original face landmarks.
    """
    input = transform(img).unsqueeze(0)  # @ReservedAssignment
    with torch.no_grad():
        if mode == 'gpu':
            input = input.cuda() # @ReservedAssignment
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
    landmarks, factor = predict_landmarks(param, bbox, dense=dense)
    pitch, yaw, roll, scale, rotation, t3d, cam_matrix = extract_pose_from_param(param)  
    return pitch, yaw, roll, scale, rotation, t3d, cam_matrix, landmarks, factor

def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """ See https://github.com/cleardusk/3DDFA.
    """
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = parse_param(param)

    if dense:
        vertices = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
        
        if transform:
            vertices[1, :] = std_size + 1 - vertices[1, :]
    else:
        vertices = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            vertices[1, :] = std_size + 1 - vertices[1, :]

    return vertices

def render_colors(vertices, triangles, colors, width, height, channels=3, bg=None):
    """ Adds color to 3D face vertices.
        See https://github.com/YadiraF/face3d.
    """
    if bg is None:
        img = np.zeros((height, width, channels), dtype = np.float32)
    else:
        assert bg.shape[0] == height and bg.shape[1] == width and bg.shape[2] == channels
        img = bg
        
    depth_buffer = np.zeros([height, width], dtype = np.float32, order = 'C') - 999999.

    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()

    mcc.render_colors_core(img, vertices, triangles, colors, depth_buffer, vertices.shape[0], triangles.shape[0], height, width, channels)  # @UndefinedVariable
    
    return img

def rotation_matrix_to_euler_angles_2(rotation):
    """ Computes three Euler angles from rotation matrix.
     
    Parameters:
        rotation (ndarray): The rotation matrix.
     
    Returns:
        float: The yaw value.
        float: The pitch value.
        float: The roll value.
    """
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] +  rotation[1, 0] * rotation[1, 0])
       
    singular = sy < 1e-6
   
    if  not singular :
        x = math.atan2(rotation[2, 1] , rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else :
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0
          
    yaw, pitch, roll = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
      
    return yaw, pitch, roll

def rotation_matrix_to_euler_angles(rotation):
    """ Computes three Euler angles from rotation matrix.
     
    Parameters:
        rotation (ndarray): The rotation matrix.
     
    Returns:
        float: The yaw value.
        float: The pitch value.
        float: The roll value.
    """
    if rotation[2, 0] != 1 or rotation[2, 0] != -1:
        x = math.asin(rotation[2, 0])
        y = math.atan2(rotation[2, 1] / math.cos(x), rotation[2, 2] / math.cos(x))
        z = math.atan2(rotation[1, 0] / math.cos(x), rotation[0, 0] / math.cos(x))
    else:
        z = 0
        if rotation[2, 0] == -1:
            x = np.pi / 2
            y = z + math.atan2(rotation[0, 1], rotation[0, 2])
        else:
            x = -np.pi / 2
            y = -z + math.atan2(-rotation[0, 1], -rotation[0, 2])

    return x, y, z

def similarity_transform(vertices, rotation, translation, scale):
    """ Calculates the new vertices using the pose information.
    
    Parameters:
        vertices (ndarray): The 3DMM vertices.
        rotation (ndarray): The rotation matrix.
        translation (ndarray): The translation vector.
        scale (float): The 3DMM scale.
    Returns:
        ndarray: The new 3DMM vertices.
    """
    
#      if dense:
#         vertices = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
# 
#         if transform:
#             vertices[1, :] = std_size + 1 - vertices[1, :]
    
    translation = np.squeeze(np.array(translation, dtype = np.float32))
    transform_vertices = scale * vertices.dot(rotation) + translation[np.newaxis, :]
    return transform_vertices
        