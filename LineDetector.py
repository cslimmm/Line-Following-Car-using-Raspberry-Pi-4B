import cv2
import numpy as np
import RPi.GPIO as gpio
import time

#Yellow
lower_range1=np.array([20,100,100])
upper_range1=np.array([40,255,255])

#Green
lower_range1=np.array([36, 118, 64])
upper_range1=np.array([81,255,153])

#Red
lower_range1=np.array([0,40,50])
upper_range1=np.array([10,255,255])

#Blue
lower_range=np.array([94,50,5])
upper_range=np.array([126,255,255])

#Black
lower_black=np.array([0, 0, 0])
upper_black=np.array([179,255,56])

class LineDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 192)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 112)

        # GPIO setup
        self.rb = 11  # right backward
        self.rf = 13  # right forward
        self.lf = 15  # left forward
        self.lb = 16  # left backward

        self.ENA = 18
        self.ENB = 19

        gpio.setwarnings(False)
        gpio.setmode(gpio.BOARD)

        gpio.setup(self.rb, gpio.OUT)
        gpio.setup(self.rf, gpio.OUT)
        gpio.setup(self.lf, gpio.OUT)
        gpio.setup(self.lb, gpio.OUT)
        gpio.setup(self.ENA, gpio.OUT)
        gpio.setup(self.ENB, gpio.OUT)

        self.pwm_left = gpio.PWM(self.ENB, 100)
        self.pwm_right = gpio.PWM(self.ENA, 100)

        self.pwm_left.start(0)
        self.pwm_right.start(0)
        gpio.output(self.rf, False)
        gpio.output(self.lf, False)
        gpio.output(self.rb, False)
        gpio.output(self.lb, False)

    def run_detection(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.detect_lines(frame)

    def detect_lines(self, frame):
        hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        blur=cv2.GaussianBlur(hsv,(5,5),0)
        red_mask = cv2.inRange(blur, lower_range, upper_range)
        green_mask = cv2.inRange(blur, lower_range1, upper_range1)
        black_mask = cv2.inRange(blur, lower_black, upper_black)
        cv2.imshow('black',black_mask)

        red_cnts, _ = cv2.findContours(red_mask, 1, cv2.CHAIN_APPROX_NONE)
        green_cnts, _ = cv2.findContours(green_mask, 1, cv2.CHAIN_APPROX_NONE)
        black_cnts, _ = cv2.findContours(black_mask, 1, cv2.CHAIN_APPROX_NONE)

        if red_cnts:
            self.contour_detection(red_cnts, frame)
        elif green_cnts:
            self.contour_detection(green_cnts, frame)
        else:
            self.contour_detection(black_cnts, frame)

    def contour_detection(self,contours, frame):
        if contours:
            for c in contours:
                if cv2.contourArea(c) > 500 and cv2.contourArea(c) < 100000:
                    M = cv2.moments(c)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    cv2.line(frame, (cx, 0), (cx, 720), (255, 0, 0), 1)
                    cv2.line(frame, (0, cy), (1280, cy), (255, 0, 0), 1)

                    if cx >= 135:
                        self.right()                        

                    if 95 < cx < 135:
                        self.forward()
                    
                    if cx <= 95:
                        self.left()
                        
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)         

        else:
            self.backward()
                        
        cv2.imshow("Line detection",frame)
            
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            self.stop()

    def forward(self):
        gpio.output(self.rf, True)
        gpio.output(self.lf, True)
        gpio.output(self.rb, False)
        gpio.output(self.lb, False)

        self.pwm_left.ChangeDutyCycle(17)
        self.pwm_right.ChangeDutyCycle(17)

    def right(self):
        gpio.output(self.rf, False)
        gpio.output(self.lf, True)
        gpio.output(self.rb, True)
        gpio.output(self.lb, False)

        self.pwm_left.ChangeDutyCycle(52)
        self.pwm_right.ChangeDutyCycle(40)

    def left(self):
        gpio.output(self.rf, True)
        gpio.output(self.lf, False)
        gpio.output(self.rb, False)
        gpio.output(self.lb, True)

        self.pwm_left.ChangeDutyCycle(40)
        self.pwm_right.ChangeDutyCycle(52)

    def backward(self):
        gpio.output(self.rf, False)
        gpio.output(self.lf, False)
        gpio.output(self.rb, True)
        gpio.output(self.lb, True)

        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(45)

    def stop(self):
        gpio.output(self.rf, False)
        gpio.output(self.lf, False)
        gpio.output(self.rb, False)
        gpio.output(self.lb, False)

        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)

    def cleanup(self):
        gpio.cleanup()
        self.cap.release()

