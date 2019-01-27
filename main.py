"""
Run this class to see 3D face tracking in real-time.
"""

import cv2
import time

import face_tracking as ft

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 30)
    font_scale = 1
    font_color = (255,255,255)
    line_type = 2
    
    _, frame = cap.read()
         
    frames_count = 0
    start = time.time()
    
    while(True): 
        _, frame = cap.read()
        
        frames_count += 1
         
        end = time.time()    
        fps  = frames_count / (end - start)
        
        cv2.putText(frame, 'FPS: ' + str(int(fps)), bottom_left_corner_of_text, font, font_scale, font_color, line_type)
        
        img = ft.track_faces(frame, show_predicted_lmks=True, show_face_fits=True)
        
        cv2.imshow('source', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    