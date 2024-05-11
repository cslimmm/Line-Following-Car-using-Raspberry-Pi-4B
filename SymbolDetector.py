import cv2
import os
import time
import RPi.GPIO as gpio
import threading

class SymbolDetector:
           
    def __init__(self):
        # GPIO setup
        self.rb = 11  # right backward
        self.rf = 13  # right forward
        self.lf = 15  # left forward
        self.lb = 16  # left backward

        gpio.setwarnings(False)
        gpio.setmode(gpio.BOARD)

        gpio.setup(self.rb, gpio.OUT)
        gpio.setup(self.rf, gpio.OUT)
        gpio.setup(self.lf, gpio.OUT)
        gpio.setup(self.lb, gpio.OUT)
        
        gpio.output(self.rf, False)
        gpio.output(self.lf, False)
        gpio.output(self.rb, False)
        gpio.output(self.lb, False)
        
        self.distance=1

    def detect(self, template_images, height,width,names,image):
        method = cv2.TM_CCOEFF_NORMED
        frame=image.copy()        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        shape_name=''

        for template, name, h,w in zip(template_images, names,height,width):

            result = cv2.matchTemplate(gray, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            location = max_loc

            if name=='Face Recognition':
                if max_val > 0.7:
                    shape_name = name  # Extract shape name from template file name
                    print('----------',shape_name)
                    cv2.putText(image, shape_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    bottom_right = (location[0] + w, location[1] + h)    
                    cv2.rectangle(image, location, bottom_right, 255, 5)
                    
            elif name== 'Calculate Distance':
                if max_val > 0.51:
                    shape_name = name
                    print('----------',shape_name)
                    cv2.putText(image, shape_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    bottom_right = (location[0] + w, location[1] + h)    
                    cv2.rectangle(image, location, bottom_right, 255, 5)
                        
            elif name=='Traffic Light':
                if max_val > 0.44:
                    start_time=time.time()
                    shape_name = name  # Extract shape name from template file name
                    print('----------',shape_name)
                    cv2.putText(image, shape_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    bottom_right = (location[0] + w, location[1] + h)    
                    cv2.rectangle(image, location, bottom_right, 255, 5)
                        
                    while (time.time()-start_time)<=5:
                        gpio.output(self.rf, False)
                        gpio.output(self.lf, False)
                        gpio.output(self.rb, False)
                        gpio.output(self.lb, False)
                        
            elif name=='No Entry':
                if max_val > 0.6:
                    elapsed_time=time.time()
                    shape_name = name  # Extract shape name from template file name
                    print('----------',shape_name)
                    cv2.putText(image, shape_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    bottom_right = (location[0] + w, location[1] + h)    
                    cv2.rectangle(image, location, bottom_right, 255, 5)
                        
                    while (time.time()-elapsed_time)<=5:
                        gpio.output(self.rf, False)
                        gpio.output(self.lf, False)
                        gpio.output(self.rb, False)
                        gpio.output(self.lb, False)
                            
        cv2.imshow('Match1', image)
        return shape_name

    def preprocess_template(self, image):
        template_files = ["stop.jpg", "faceReco4.jpg", "distance1.jpg","noEntry2.jpg"]
        templates = [cv2.imread(os.path.join("/home/pi/Downloads/Symbols", file)) for file in template_files]
        names = ["Traffic Light", "Face Recognition", "Calculate Distance","No Entry"]
        
        # Load and preprocess templates
        template_images = []
        height = []
        width = []
            
        for template in templates:
            template = cv2.resize(template, (0, 0), fx=0.75, fy=0.75)
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            h, w , _= template.shape

            template_images.append(gray)
            height.append(h)
            width.append(w)
        return template_images, height,width,names
  

