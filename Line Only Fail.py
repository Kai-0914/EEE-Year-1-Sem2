import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import time
import math

IN1 = 11
IN2 = 13
IN3 = 15
IN4 = 12
ENA = 16 
ENB = 18 

GPIO.setmode(GPIO.BOARD)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwmA = GPIO.PWM(ENA, 255)
pwmB = GPIO.PWM(ENB, 255)
base_speed = 25
pwmA.start(base_speed)  
pwmB.start(base_speed)

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    image_center_x = 640//2
    
    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret_thresh, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                M = cv2.moments(largest_contour)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    
                    GPIO.output(IN1, GPIO.LOW)
                    GPIO.output(IN2, GPIO.HIGH)
                    GPIO.output(IN3, GPIO.LOW)
                    GPIO.output(IN4, GPIO.HIGH)
                    
                    error = cx - image_center_x
                    print(f"Centroid: ({cx}, {cy}), Error: {error}")
                    
                    if abs(error) < 20:
                        error = 0
                    else:
                        adjustment = 0.25 * error
                        left_speed = base_speed - adjustment
                        right_speed = base_speed + adjustment
                        left_speed = max(0, min(100, left_speed))
                        right_speed = max(0, min(100, right_speed))
                        pwmA.ChangeDutyCycle(left_speed)
                        pwmB.ChangeDutyCycle(right_speed)
                
            else:
                GPIO.output(IN1, GPIO.LOW)
                GPIO.output(IN2, GPIO.LOW)
                GPIO.output(IN3, GPIO.LOW)
                GPIO.output(IN4, GPIO.LOW)                

            cv2.imshow("Frame", frame)
            cv2.imshow("Threshold", thresh)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
