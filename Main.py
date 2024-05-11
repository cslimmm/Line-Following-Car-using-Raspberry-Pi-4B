import cv2
import threading
import numpy as np
import LineDetector
import ArrowDetector
import ShapeDetector
import SymbolDetector3
import FaceDetector
import RPi.GPIO as gpio

#import distanceCalculation
import time
import ctypes
from ctypes.util import find_library

# Load the Xlib library
xlib = ctypes.cdll.LoadLibrary(find_library('X11'))

# Call XInitThreads
xlib.XInitThreads()

start_time=time.time()

encoderA = 21  # left
encoderB = 22  # right
stateCount = 0
CIRCUMFERENCE = 22
STATES_PER_REVOLUTION = 40
distance=0

gpio.setmode(gpio.BOARD)

gpio.setup(encoderA, gpio.IN, pull_up_down=gpio.PUD_UP)
gpio.setup(encoderB, gpio.IN, pull_up_down=gpio.PUD_UP)

last_stateA = gpio.input(encoderA)
last_stateB = gpio.input(encoderB)

def line_detection_thread(line_detector):
    line_detector.run_detection()

def main():
    global last_stateA,last_stateB, stateCount,CIRCUMFERENCE,STATES_PER_REVOLUTION,distance
    flag=1
    load=1
    
    line_detector = LineDetector.LineDetector()
    arrow_detector = ArrowDetector.ArrowDetector()
    shape_detector = ShapeDetector.ShapeDetector()
    symbol_detector = SymbolDetector3.SymbolDetector()
    face_recognizer = FaceDetector.Face_Recog()
        
    face_thread = None
    
    while True:
        ret, image = line_detector.cap.read()
        
        if not ret:
            break
        
        # Start LineDetector in a separate thread
        line_thread = threading.Thread(target=line_detector.run_detection)
        line_thread.start()
    
        frame=image.copy()
       
        if load==1:
            template_images, height,width,names = symbol_detector.preprocess_template(image.copy())
            load=0
        
        symbol=symbol_detector.detect(template_images, height,width,names,image.copy())
        direction = arrow_detector.detect_arrow_direction(image.copy())
        detected_shapes = shape_detector.detect_shapes(image.copy())         

        if symbol == "Calculate Distance":
            flag=0
            
        if flag==0:
            current_stateA = gpio.input(encoderA)

            if current_stateA != last_stateA:
                last_stateA = current_stateA
                stateCount += 1
                
            current_stateB = gpio.input(encoderB)

            if current_stateB != last_stateB:
                last_stateB = current_stateB
                stateCount += 1

            distance = CIRCUMFERENCE * stateCount / STATES_PER_REVOLUTION
            print('---------------------distance =', distance)

        if symbol == "Face Recognition":
            flag=1

            if face_thread is None or not face_thread.is_alive():
                print("Distance =",distance)
                while True:
                    ret, image = line_detector.cap.read()
                    face_thread = threading.Thread(target=face_recognizer.face_recog, args=(image,))
                    face_thread.start()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    line_detector.cap.release()
    line_detector.cleanup() 
    if face_thread and face_thread.is_alive():
        face_thread.join()

if __name__ == "__main__":
    main()
