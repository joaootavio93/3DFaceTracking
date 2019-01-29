"""
Run this class to save the 3D face tracking pipeline into a folder.
"""

import cv2
import os
import time
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
        print('Processing ' + output_name + ':')
        
        start = time.time()
        bboxes = fd.detect_faces(img, detector=fd.Face_Detector.DLIB)
        elapsed_time_1 = time.time() - start
        print('\tface detection: %.5f sec' % elapsed_time_1)
        
        if len(bboxes):    
            bboxes_img = fd.draw_bboxes(img.copy(), bboxes)
            cv2.imwrite(output_folder + output_name + '_1.jpg', bboxes_img)
            
            start = time.time()
            new_bboxes = fd.resize_bboxes(bboxes)
            elapsed_time_2 = time.time() - start
            print('\tbbox resizing: %.5f sec' % elapsed_time_2)
                      
            new_bboxes_img = fd.draw_bboxes(img.copy(), new_bboxes)
            cv2.imwrite(output_folder + output_name + '_2.jpg', new_bboxes_img)
            
            start = time.time()
            crops = fd.crop_faces(img, new_bboxes)
            elapsed_time_3 = time.time() - start
            print('\tface cropping: %.5f sec' % elapsed_time_3)
            
            count_crops = 0
            
            for crop_img in crops:
                print('\t\tCrop ' + str(count_crops) + ':')
                
                crop_h, crop_w, _ = crop_img.shape
                cv2.imwrite(output_folder + output_name + '_3_' + str(count_crops) + '.jpg', crop_img)
                
                start = time.time()
                new_img = utils.resize_image(crop_img.copy(), width=fa.std_size, height=fa.std_size)
                elapsed_time_4 = time.time() - start
                print('\t\t\tcrop resizing: %.5f sec' % elapsed_time_4)
                
                cv2.imwrite(output_folder + output_name + '_4_' + str(count_crops) + '.jpg', new_img)
                
                start = time.time()
                pitch, yaw, roll, scale, rotation, t3d, cam_matrix, landmarks, factor = fa.predict_pose(new_img, new_bboxes[count_crops], dense=False)
                elapsed_time_5 = time.time() - start
                print('\t\t\tpose prediction: %.5f sec' % elapsed_time_5)

                lmks_image = fd.draw_landmarks(img.copy(), landmarks, drawer=fd.Face_Landmarks_Drawer.DDFA)
                cv2.imwrite(output_folder + output_name + '_5_' + str(count_crops) + '.jpg', lmks_image)
                
                start = time.time()
                fit_img, fit_kpts = ff.fit_3dmm(rotation, t3d, scale * factor, width=crop_w, height=crop_h)
                elapsed_time_6 = time.time() - start
                print('\t\t\tface fitting: %.5f sec' % elapsed_time_6)
                
                cv2.imwrite(output_folder + output_name + '_6_' + str(count_crops) + '.png', fit_img)
                        
                fit_lmks_img = fd.draw_landmarks(fit_img.copy(), fit_kpts, drawer=fd.Face_Landmarks_Drawer.FACE3D)
                cv2.imwrite(output_folder + output_name + '_7_' + str(count_crops) + '.jpg', fit_lmks_img)
                
                start = time.time()
                fore_lmks, back_lmks = utils.landmarks_to_np_array(fit_kpts, landmarks)
                elapsed_time_7 = time.time() - start
                print('\t\t\tlandmarks setting: %.5f sec' % elapsed_time_7)
                
                start = time.time()
                homography, mask = cv2.findHomography(fore_lmks, back_lmks, cv2.RANSAC)
                elapsed_time_8 = time.time() - start
                print('\t\t\thomography computation: %.5f sec' % elapsed_time_8)
                
                start = time.time()
                warp_img = cv2.warpPerspective(fit_img, homography, (w, h))
                elapsed_time_9 = time.time() - start
                print('\t\t\twarp perspective: %.5f sec' % elapsed_time_9)
                
                cv2.imwrite(output_folder + output_name + '_8_' + str(count_crops) + '.png', warp_img)
                
                start = time.time()
                blend_img = utils.create_transparent_overlay_faster(warp_img, img, w, h)
                elapsed_time_10 = time.time() - start
                print('\t\t\ttransparent overlay: %.5f sec' % elapsed_time_10)
                
                cv2.imwrite(output_folder + output_name + '_9_' + str(count_crops) + '.jpg', blend_img)
  
                count_crops += 1
                
                print('\t\t\telapsed time: %.5f' % (elapsed_time_1 + elapsed_time_2 + elapsed_time_3 + elapsed_time_4 + elapsed_time_5 + elapsed_time_6
                      + elapsed_time_7 + elapsed_time_8 + elapsed_time_9 + elapsed_time_10))
                print('\tfitted succefully!')
            
    count += 1
