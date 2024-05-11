import numpy as np
import cv2 as cv
import picamera
import time
from picamera.array import PiRGBArray
from images_stack import *
from threshold_trackbars import *
import math

def combined_function():
    # Initialize camera
    camera = picamera.PiCamera()
    camera.resolution = (192, 112)
    camera.framerate = 20
    rawCapture = PiRGBArray(camera, size=(192, 112))
    time.sleep(0.1)

    threshold_trackbars_create()

    final_text = "-----"
    text_counter = 0

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array

        blur, b, c, area, epsilon = threshold_trackbars_pos()

        frame = cv.resize(frame, (500, 500))
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur = cv.medianBlur(frame_grey, blur)
        frame_threshold = cv.adaptiveThreshold(frame_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, b, c)

        font_face = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        contours, _ = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour_index in range(len(contours)):
            if cv.contourArea(contours[contour_index], True) > area:
                perimeter = cv.arcLength(contours[contour_index], True)
                poly = cv.approxPolyDP(contours[contour_index], epsilon * perimeter, True)
                bbox_x, bbox_y, bbox_w, bbox_h = cv.boundingRect(poly)
                cv.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 0), 2)


                num_circles = 0
                num_triangles = 0
                num_rectangles = 0
                num_squares = 0
                num_hexagon = 0
                num_pentagon = 0
                num_partialCircle = 0
                
                num_corners = len(poly)
                
                if num_corners == 3:
                    cv.putText(frame, "Triangle", (bbox_x, bbox_y - 10), font_face, font_scale, (0, 0, 0), 1)
                    num_triangles += 1
                    break
                
                if num_corners == 4:
                    cv.putText(frame, "Rectangle", (bbox_x, bbox_y - 10), font_face, font_scale, (0, 0, 0), 1)
                    num_rectangles += 1
                    break
                
                elif num_corners == 6:
                    cv.putText(frame, "Hexagon", (bbox_x, bbox_y - 10), font_face, font_scale, (0, 0, 0), 1)
                    num_hexagon += 1
                    break
            
                elif num_corners == 5:
                    cv.putText(frame, "Pentagon", (bbox_x, bbox_y - 10), font_face, font_scale, (0, 0, 0), 1)
                    num_pentagon += 1
                    break
                    
                else:
                    corners = np.int0(poly)
                    for i in corners:
                        x,y = i.ravel()
                        cv.circle(frame,(x,y),10,(0,0,255),-1)
                      
                    am = (corners[0][0][0] + corners[1][0][0]) / 2
                    bm = (corners[0][0][1] + corners[1][0][1]) / 2
                    cv.circle(frame,  (int(am), int(bm)), 5, (255, 0, 0), -1)

                    cont, _ = cv.findContours(frame_blur.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    cont = sorted(cont, key=lambda x: cv.contourArea(x), reverse=True)

                    if len(cont[0]) > 0:
                        (x, y), radius = cv.minEnclosingCircle(cont[0])
                        cv.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 3)
                        cv.line(frame, (int(x), int(y)), (int(am),int(bm)),(255,0,0),2)
                        cv.line(frame, (int(x), int(y)), (int(radius+x),int(y)),(255,255,0),2)

                        atan = math.atan2(int(bm)-int(y),int(am)-int(x))
                        angle = math.degrees(atan)
                        
                        (x, y), radius = cv.minEnclosingCircle(contours[contour_index])
                        area_ratio = cv.contourArea(contours[contour_index]) / (np.pi * (radius ** 2))
                        if area_ratio <= 0.5:  
                            cv.putText(frame, "Partial Circle", (bbox_x, bbox_y - 10), font_face, font_scale, (0, 0, 0), 1)
                            num_partialCircle += 1
                            
                        else:
                            cv.putText(frame, "Circle", (bbox_x, bbox_y - 10), font_face, font_scale, (0, 0, 0), 1)
                            num_circles += 1
                        
                        if angle >= -45 and angle < 45:
                            arrow_text = "RIGHT"
                            cv.putText(frame,arrow_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)
                            break
                        elif angle >=45 and angle < 135:
                            arrow_text = "DOWN"
                            cv.putText(frame,arrow_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)
                            break
                        elif angle >= -180 and angle <=-135: 
                            arrow_text = "LEFT"
                            cv.putText(frame,arrow_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)
                            break
                        elif angle >=135 and angle <=180:
                            arrow_text = "LEFT"
                            cv.putText(frame,arrow_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)
                            break
                        elif angle > -135 and angle < -45:
                            arrow_text = "UP"
                            cv.putText(frame,arrow_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)
                            break
                        
                        if final_text != arrow_text:
                            text_counter += 1
                        else:
                            text_counter = 0
                    
                        if text_counter >= 5 :
                            final_text = arrow_text
                        
                    break
                        

        cv.putText(frame, f"Triangles: {num_triangles}", (20, 20), font_face, font_scale, (0, 0, 0), 1)
        cv.putText(frame, f"Rectangles: {num_rectangles}", (20, 40), font_face, font_scale, (0, 0, 0), 1)
        cv.putText(frame, f"Squares: {num_squares}", (20, 60), font_face, font_scale, (0, 0, 0), 1)
        cv.putText(frame, f"Circles: {num_circles}", (20, 80), font_face, font_scale, (0, 0, 0), 1)
        cv.putText(frame, f"Partial Circles: {num_partialCircle}", (20, 100), font_face, font_scale, (0, 0, 0), 1)
        cv.putText(frame, f"Pentagon: {num_pentagon}", (20, 120), font_face, font_scale, (0, 0, 0), 1)
        cv.putText(frame, f"Hexagon: {num_hexagon}", (20, 140), font_face, font_scale, (0, 0, 0), 1)

        stack = stack_images(1, [[frame, frame_grey], [frame_blur, frame_threshold]])
        cv.imshow("frame", stack)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
        rawCapture.truncate(0)

    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()
                                                                                       
combined_function()
