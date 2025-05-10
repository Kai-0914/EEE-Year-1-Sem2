import cv2
import numpy as np
import time
import math
import RPi.GPIO as GPIO
from picamera2 import Picamera2

IN1 = 11
IN2 = 13
IN3 = 15
IN4 = 12
ENA = 16
ENB = 18

# GPIO setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# PWM setup
pwmA = GPIO.PWM(ENA, 255)
pwmB = GPIO.PWM(ENB, 255)
base_speed = 15
pwmA.start(base_speed)
pwmB.start(base_speed)

Kp = 0.5   # Proportional gain
Ki = Kp * 0.01  # Integral gain
Kd = Kp * 0.1  # Derivative gain
prev_error = 0
integral = 0
last_time = time.time()
MAX_ADJUSTMENT = 40  # Limits steering aggression
previous_line_type = None

colors = [
    {   # Blue
        "lower": np.array([100, 50, 50]),
        "upper": np.array([130, 255, 255]),
        "name": "Blue",
        "color": (255, 0, 0)
    },
    {   # Green
        "lower": np.array([40, 50, 50]),
        "upper": np.array([80, 255, 255]),
        "name": "Green",
        "color": (0, 255, 0)
    },
    {   # Red
        "lower": [np.array([0, 50, 50]), np.array([160, 50, 50])],
        "upper": [np.array([10, 255, 255]), np.array([180, 255, 255])],
        "name": "Red",
        "color": (0, 0, 255)
    },
    {   # Yellow
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255]),
        "name": "Yellow",
        "color": (139, 128, 0)
    }
]

PRIORITY_COLORS = ['Red', 'Green']

def get_color_info(color_name):
    for color in colors:
        if color['name'] == color_name:
            return color
    return None

priority_colors = [get_color_info(name) for name in PRIORITY_COLORS]

resolution_x = 640
resolution_y = 480

def shapes(frame):
    # 1) Mask out all dark (line) pixels in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([150, 50, 80])   # allow a bit of S/V for dark gray
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    inv_mask = cv2.bitwise_not(black_mask)
    filtered = cv2.bitwise_and(frame, frame, mask=inv_mask)

    # 2) Standard gray→blur→binary_inv
    gray    = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)

    # 3) Find contours & drop any that touch the bottom edge (the guide line)
    h_frame, w_frame = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    keep = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # drop if it touches the bottom of the ROI
        if y + h >= h_frame - 1:
            continue
        # drop if it spans almost entire width
        if w > 0.8 * w_frame:
            continue
        keep.append(c)

    if not keep:
        return frame

    # 4) Pick the largest valid contour, draw, approximate & label
    best = max(keep, key=cv2.contourArea)
    cv2.drawContours(frame, [best], -1, (0,255,0), 2)

    peri   = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    v      = len(approx)
    shape  = None 
    if   v == 3: shape = "triangle"
    elif v == 4: shape = "rectangle"
    elif v == 5: shape = "pentagon"
    elif v == 6: shape = "hexagon"
    else:
        circ = (4 * math.pi * cv2.contourArea(best)) / (peri*peri)
        if circ > 0.8:
            shape = "circle"
        else:
            shape = "3/4 circle"

    M = cv2.moments(best)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cv2.putText(frame, shape, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        print(f"{shape}")


    return frame



picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (resolution_x, resolution_y)})
picam2.configure(config)
picam2.start()

SHAPE_COOLDOWN = 3
last_shape_time = 0

crop_height = 70
center_x = resolution_x // 2

try:
    while True:
        raw_frame = picam2.capture_array()
        
        if raw_frame.shape[2] == 4:
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            
        line_raw = frame[-crop_height:, :]
        symbol_raw = frame[:-crop_height, :]
        
        line = line_raw.copy()
        hsv = cv2.cvtColor(line_raw, cv2.COLOR_BGR2HSV)
        symbol = shapes(symbol_raw.copy())
        
        detected_color = None
        target_cx_color = None
        
        for color in priority_colors:
            if color is None:
                continue
            
            if color['name'] == 'Red':
                mask1 = cv2.inRange(hsv, color['lower'][0], color['upper'][0])
                mask2 = cv2.inRange(hsv, color['lower'][1], color['upper'][1])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color['lower'], color['upper'])

            mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 500:
                continue

            M = cv2.moments(largest)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            detected_color = color
            cx_color = cx

            cv2.drawContours(line, [largest], -1, color['color'], 2)
            cv2.circle(line, (cx, cy), 7, color['color'], -1)
            cv2.putText(line, f"{color['name']} ({cx},{cy})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color['color'], 2)
            break
        
        if detected_color:
            current_line_type = detected_color['name']  
        else:
            current_line_type = 'Black' 
            
        if current_line_type != previous_line_type:
            integral = 0
            prev_error = 0
            previous_line_type = current_line_type

        if detected_color is None:
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 100])
            
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            kernel = np.ones((5,5), np.uint8)
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN,  kernel)
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
            black_line_detected = False
            
            if contours:
                largest = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest) > 500:
                    M = cv2.moments(largest)
                    
                    if M["m00"] != 0:
                        black_line_detected = True
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(line, [largest], -1, (0,255,0), 2)
                        cv2.circle(line, (cx, cy), 7, (0,0,255), -1)
    
                        current_time = time.time()
                        dt = current_time - last_time
                        error = cx - center_x
                        integral += error * dt
                        derivative = (error - prev_error) / dt
                        adjustment = (Kp * error) + (Ki * integral) + (Kd * derivative)
                        integral = max(min(integral, 1000), -1000)
                        adjustment = max(min(adjustment, MAX_ADJUSTMENT), -MAX_ADJUSTMENT)
                        prev_error = error
                        last_time = current_time
                        
                        left_speed  = base_speed - adjustment
                        right_speed = base_speed + adjustment
                        
                        left_speed = max(0, min(100, left_speed))
                        right_speed = max(0, min(100, right_speed))
                        
                        GPIO.output(IN1, GPIO.LOW)
                        GPIO.output(IN2, GPIO.HIGH)
                        GPIO.output(IN3, GPIO.LOW)
                        GPIO.output(IN4, GPIO.HIGH)
                        
                        pwmA.ChangeDutyCycle(left_speed)
                        pwmB.ChangeDutyCycle(right_speed)

        if detected_color is not None:
            error = cx_color - center_x
            adjustment = 0.25 * error
            left_speed  = base_speed - adjustment
            right_speed = base_speed + adjustment
            
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)
            
            pwmA.ChangeDutyCycle(max(0,min(100,left_speed)))
            pwmB.ChangeDutyCycle(max(0,min(100,right_speed)))
                
        if detected_color is None and not black_line_detected:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
            pwmA.ChangeDutyCycle(50)
            pwmB.ChangeDutyCycle(50)

        cv2.imshow("Line Following with Color", line)
        cv2.imshow("symbol detection", symbol)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
