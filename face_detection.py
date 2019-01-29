"""
Contains functions for face detection purposes.
Part of this code is based on Jianzhu Guo's code (https://github.com/cleardusk/3DDFA).
"""

import cv2
import dlib

import numpy as np

#from faced import FaceDetector

dlib_detector = dlib.get_frontal_face_detector()
#faced_detector, faced_thresh = FaceDetector(), 0.5

class Bbox_Resizer:
    """
    Represents a face bounding box resizer algorithm.
    """
    DDFA = 0

class Face_Cropper:
    """
    Represents a face bounding box cropper algorithm.
    """
    DDFA = 0

class Face_Detector:
    """
    Represents a face detector algorithm.
    """
    FACED = 0,
    DLIB = 1
    
class Face_Landmarks_Drawer:
    """
    Represents a face landmarks painter algorithm.
    """
    DDFA = 0,
    FACE3D = 1

def crop_faces(img, bboxes, cropper=Face_Cropper.DDFA):
    """ Crops the detected faces from an image.
    
    Parameters:
        img (mat): The source image.
        bboxes (list): The detected face bounding boxes.
        cropper (class): The face cropper algorithm.
        
    Returns:
        list(mat): The cropped images. 
    """
    crops = []
    if cropper == Face_Cropper.DDFA:
        h, w = img.shape[:2]
        
        for bbox in bboxes:
            sx, sy, ex, ey = [int(round(_)) for _ in bbox]
            dh, dw = ey - sy, ex - sx
            if len(img.shape) == 3:
                res = np.zeros((dh, dw, 3), dtype=np.uint8)
            else:
                res = np.zeros((dh, dw), dtype=np.uint8)
            if sx < 0:
                sx, dsx = 0, -sx
            else:
                dsx = 0
        
            if ex > w:
                ex, dex = w, dw - (ex - w)
            else:
                dex = dw
        
            if sy < 0:
                sy, dsy = 0, -sy
            else:
                dsy = 0
        
            if ey > h:
                ey, dey = h, dh - (ey - h)
            else:
                dey = dh
        
            res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
            
            crops.append(res)
    return crops

def detect_faces(img, detector=Face_Detector.DLIB):
    """ Detects the faces in an image.
    
    Parameters:
        img (mat): The source image.
        detector (class): The face detector algorithm.
    Returns:
        list: The face bounding boxes.
    """
    bboxes = []
    faces = []
    
    if detector == Face_Detector.FACED:
        rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        faces = faced_detector.predict(rgb_img, faced_thresh)
    elif detector == Face_Detector.DLIB:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dlib_detector(gray_img, 1)
    
    for face in faces:
        if detector == Face_Detector.FACED:
            x, y, w, h, prob = face
            if prob > 0.75:
                bboxes.append([int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)])
        elif detector == Face_Detector.DLIB:
            bboxes.append([face.left(), face.top(), face.right(), face.bottom()])
                
    return bboxes

def draw_bboxes(img, bboxes):
    """ Draws the face bounding boxes on an image.
    
    Parameters:
        img (mat): The source image.
        bboxes (list): The face bounding boxes.
    Returns:
        list: The image with drawn face bounding boxes.
    """
    for bbox in bboxes:
        draw_bbox(img, bbox)
    return img

def draw_bbox(img, bbox):
    """ Draws the face bounding box on an image.
    
    Parameters:
        img (mat): The source image.
        bboxes (list): The face bounding box.
    Returns:
        list: The image with drawn face bounding box.
    """
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=2)
    return img

def draw_landmarks(img, landmarks, drawer=Face_Landmarks_Drawer.DDFA):
    """ Draws the faces landmarks on an image.
    
    Parameters:
        img (mat): The source image.
        landmarks (list): The faces landmarks.
        drawer (class): The landmarks drawer algorithm.
    Returns:
        mat: The image with drawn landmarks.
    """
    if drawer==Face_Landmarks_Drawer.DDFA and len(landmarks) and landmarks[0] is not None and landmarks[1] is not None:
        lmks_x, lmks_y = landmarks[0].T, landmarks[1].T   
        for i in range(len(lmks_x)):
            cv2.circle(img, (int(lmks_x[i]), int(lmks_y[i])), 3, (255, 255, 255), -1)
    elif drawer==Face_Landmarks_Drawer.FACE3D and landmarks is not None:
        for x, y in landmarks:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    return img

def resize_bbox(bbox, resizer=Bbox_Resizer.DDFA):
    """ Resizes the face bounding box.
    
    Parameters:
        bbox (array): The face bounding box.
        resizer (class): The bounding box resizer algorithm.
    Returns:
        array: The resized bounding box.
    """
    if resizer == Bbox_Resizer.DDFA:
        left, top, right, bottom = bbox
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
        size = int(old_size * 1.58)
        bbox = [0] * 4
        bbox[0] = center_x - size / 2
        bbox[1] = center_y - size / 2
        bbox[2] = bbox[0] + size
        bbox[3] = bbox[1] + size
    return bbox

def resize_bboxes(bboxes, resizer=Bbox_Resizer.DDFA):
    """ Resizes the face bounding boxes.
    
    Parameters:
        bbox (array): The face bounding boxes.
        resizer (class): The bounding box resizer algorithm.
    Returns:
        array: The resized bounding boxes.
    """
    new_bboxes = []
    for bbox in bboxes:
        new_bboxes.append(resize_bbox(bbox, resizer))
    return new_bboxes
