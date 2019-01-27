"""
Run this class to save the 3D face tracking pipeline into a folder.
"""

import cv2
import os
import utils
import face_alignment as fa
import face_detection as fd
import face_fit as ff

input_folder = './data/source/' # The input images path.
output_folder = './data/results/' # The output images path.

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
files = os.listdir(input_folder)
count = 1

for file in files:
    output_name = 'image_' + str(count)

    img = cv2.imread(input_folder + file, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    if img is not None:
        bboxes = fd.detect_faces(img, detector=fd.Face_Detector.DLIB)
        if len(bboxes):    
            bboxes_img = fd.draw_bboxes(img.copy(), bboxes)
            cv2.imwrite(output_folder + output_name + '_1.jpg', bboxes_img)
            
            new_bboxes = fd.resize_bboxes(bboxes)
            new_bboxes_img = fd.draw_bboxes(img.copy(), new_bboxes)
            cv2.imwrite(output_folder + output_name + '_2.jpg', new_bboxes_img)
            
            crops = fd.crop_faces(img, new_bboxes)
            count_crops = 0
            
            for crop_img in crops:
                crop_h, crop_w, _ = crop_img.shape
                cv2.imwrite(output_folder + output_name + '_3_' + str(count_crops) + '.jpg', crop_img)
                
                new_img = utils.resize_image(crop_img.copy(), width=fa.std_size, height=fa.std_size)
                cv2.imwrite(output_folder + output_name + '_4_' + str(count_crops) + '.jpg', new_img)
            
                pitch, yaw, roll, scale, rotation, t3d, cam_matrix, landmarks, factor = fa.predict_pose(new_img, new_bboxes[count_crops])
                pose = pitch, yaw, roll
                
                lmks_image = fd.draw_landmarks(img.copy(), landmarks, drawer=fd.Face_Landmarks_Drawer.DDFA)
                cv2.imwrite(output_folder + output_name + '_5_' + str(count_crops) + '.jpg', lmks_image)
                
                fit_img, fit_kpts = ff.fit_3dmm(rotation, [0, 0, 0], scale * factor, width=crop_w, height=crop_h)
                cv2.imwrite(output_folder + output_name + '_6_' + str(count_crops) + '.jpg', fit_img)
                        
                fit_lmks_img = fd.draw_landmarks(fit_img.copy(), fit_kpts, drawer=fd.Face_Landmarks_Drawer.FACE3D)
                cv2.imwrite(output_folder + output_name + '_7_' + str(count_crops) + '.jpg', fit_lmks_img)
                
                count_crops += 1
                
                print(output_name + ' (' + str(count_crops) + ') fitted succefully!')
            
    count += 1
    