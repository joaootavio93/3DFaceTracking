"""
Contains functions for 3D face tracking and face augmentation for Augmented Reality purposes.
"""

import cv2
import utils
import face_alignment as fa
import face_detection as fd
import face_fit as ff

def track_faces(img):
    """ Computer 3D face tracking.
    
     Parameters:
        img (list): The input image.
        
    Returns:
        mat: A image showing the 3D face tracking.
    """
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
                pitch, yaw, roll, scale, rotation, t3d, cam_matrix, landmarks, factor = fa.predict_pose(new_img, new_bboxes[count_crops], dense=False)
                fit_img, fit_kpts = ff.fit_3dmm(rotation, t3d, scale * factor, width=crop_w, height=crop_h)                     
                fore_lmks, back_lmks = utils.landmarks_to_np_array(fit_kpts, landmarks)
                homography, mask = cv2.findHomography(fore_lmks, back_lmks, cv2.RANSAC)
                warp_img = cv2.warpPerspective(fit_img, homography, (w, h))
                img = utils.create_transparent_overlay_faster(warp_img, img, w, h)
                count_crops += 1            
    return img       