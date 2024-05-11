from picamera.array import PiRGBArray
import RPi.GPIO as gpio
import time
import cv2
import picamera
import numpy as np
import os

# Initialize camera
camera = picamera.PiCamera()
camera.resolution = (304, 208)
camera.framerate = 30  # Reduced framerate
rawCapture = PiRGBArray(camera, size=(304, 208))
time.sleep(0.1)

gpio.setwarnings(False)
gpio.setmode(gpio.BOARD)

def detect():
    templates = [
        cv2.imread(os.path.join("/home/pi/Downloads", "stop1.jpg")),
        cv2.imread(os.path.join("/home/pi/Downloads", "stop2.jpg")),
        cv2.imread(os.path.join("/home/pi/Downloads", "faceReco.jpg")),
        cv2.imread(os.path.join("/home/pi/Downloads", "distance.jpg"))
    ]
    names = [
        "Stop",
        "Stop",
        "Face Recognition",
        "Calculate Distance"
    ]

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        for template, name in zip(templates, names):
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_thresh = cv2.threshold(template_gray, 127, 255, 0)

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, image_thresh = cv2.threshold(image_gray, 127, 255, 0)

            result = cv2.matchTemplate(template_thresh, image_thresh, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > 6000000:
                rect = cv2.minAreaRect(image)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                cv2.putText(image, name, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Match', image)
        cv2.waitKey(1)
        rawCapture.truncate(0)

try:
    detect()
finally:
    gpio.cleanup()
    camera.close()
    cv2.destroyAllWindows()

