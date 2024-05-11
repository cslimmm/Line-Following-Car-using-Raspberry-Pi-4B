import cv2
import numpy as np

class ShapeDetector:
    def __init__(self):
        pass
       
    def detect_shapes(self, image):
        # Your shape detection logic here
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shapes = []
        for contour in contours:
            # Approximate the contour to a simpler polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        
            # Get the number of vertices (edges) of the polygon
            num_vertices = len(approx)
        
            # Default shape label
            shape = " "
        
            if num_vertices == 3:
                shape = "Triangle"
                print('---------Shape:',shape)
            elif num_vertices == 4:
                shape = "Rectangle"
                print('--------Shape:',shape)
            if num_vertices == 5:
                shape = "Pentagon"
                print('-----------Shape:',shape)
                        
            elif num_vertices == 6:
                shape = "Hexagon"
                print('----------Shape:',shape)
                        
            else:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
            
                if perimeter != 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circle_check = cv2.matchShapes(contour, cv2.approxPolyDP(cv2.convexHull(contour), epsilon, True), 1, 0.0)
                    if circle_check < 0.05 and area / perimeter**2 < 0.8:
                        if circularity >= 0.7:
                            shape = "Circle"
                            print('------------Shape:',shape)
                        else:
                            shape = "Partial Circle"
                            print('--------------Shape:',shape)
                else:
                    shape = "Unknown"

            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('Shape',image)

            detected_shapes.append((shape, approx))
        
        return detected_shapes

