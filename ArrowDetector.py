import cv2
import numpy as np

class ArrowDetector:
    def __init__(self):
        pass
    
    def detect_arrow_direction(self, image):
        arrow = image.copy()
        gray = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        mask = cv2.dilate(edges.copy(), None, iterations=1)
        mask = cv2.erode(mask, None, iterations=1)
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('Arrow',contours)

        internal_contours = []
        
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                # Check if contour is internal (has a parent contour)
                if hierarchy[0][i][3] != -1:
                    # Filter contours based on area
                    area = cv2.contourArea(contour)
                    if area > 100:  # Adjust threshold as needed
                        internal_contours.append(contour)
        
        for c in internal_contours:
            if len(c) > 0:
                M = cv2.moments(c)
                if M["m00"] != 0:  # Check if m00 is not zero to avoid division by zero
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                        
                    max_distance = 0
                    tip = None
                    for point in c:
                        distance = np.linalg.norm(np.array(point[0]) - np.array([centroid_x, centroid_y])) - 2
                        if distance > max_distance:
                            max_distance = distance
                            tip = point[0]
                        
                    if tip is not None:
                        cv2.circle(arrow, (tip[0], tip[1]), 3, (255, 255, 0), thickness=3)
                        cv2.circle(arrow, (centroid_x,centroid_y), 3, (255, 0, 255), thickness=3) 
                        
                    direction = ""
                    if tip is not None:
                        if tip[1] - centroid_y >= 12:
                            direction += "Down"
                        elif tip[1] - centroid_y < 12:
                            direction += "Up"
                            
                        if tip[0]-centroid_x < 0:
                            direction += "Left"
                        elif tip[0] - centroid_x >= 0:
                            direction += "Right"
                        
                        print('-----------Arrow:',direction)
                        cv2.putText(arrow, direction, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.drawContours(arrow, c, -1, (0, 255, 0), 5)
                    cv2.imshow('arrow', arrow)

                    return direction

