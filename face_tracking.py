"""
Contains functions for 3D face tracking and face augmentation for Augmented Reality purposes.
"""

import cv2
import utils
import face_alignment as fa
import face_detection as fd
import face_fit as ff
from cv2.cv2 import imshow

def track_faces(img):

    h, w, _ = img.shape
    
    if img is not None:
        bboxes = fd.detect_faces(img, detector=fd.Face_Detector.DLIB)
        if len(bboxes):    
            new_bboxes = fd.resize_bboxes(bboxes)
            
            crops = fd.crop_faces(img, new_bboxes)
            count_crops = 0
            
            for crop_img in crops:
                crop_h, crop_w, _ = crop_img.shape
                
                new_img = utils.resize_image(crop_img.copy(), width=fa.std_size, height=fa.std_size)
            
                pitch, yaw, roll, scale, rotation, t3d, cam_matrix, landmarks, factor = fa.predict_pose(new_img, new_bboxes[count_crops])
                
                fit_img, fit_kpts = ff.fit_3dmm(rotation, [0, 0, 0], scale * factor, width=crop_w, height=crop_h)
                
                imshow
            