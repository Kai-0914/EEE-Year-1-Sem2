import cv2
import numpy as np
import time
import math
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

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
base_speed = 20
pwmA.start(base_speed)
pwmB.start(base_speed)

MODEL_PATH = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'

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

PRIORITY_COLORS = ['Red', ''] 

def get_color_info(color_name):
    for color in colors:
        if color['name'] == color_name:
            return color
    return None

priority_colors = [get_color_info(name) for name in PRIORITY_COLORS]

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(LABELS_PATH)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_height = input_details[0]['shape'][1]
model_width = input_details[0]['shape'][2]

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

symbol_pause = False
pause_duration = 3
pause_start_time = 0
line_center_x = 320

try:
    while True:
        frame = picam2.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        if frame.shape[2] == 4:
            raw_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        symbol = raw_frame.copy()
        line = raw_frame.copy()

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
            cv2.putText(line, f"{color['name']} ({cx},{cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color['color'], 2)
            break

        if detected_color is None:
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            kernel = np.ones((5,5), np.uint8)
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN,  kernel)
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           
            black_line_detected = False
            
            if contours and not symbol_pause:
                largest = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest) > 500:
                    M = cv2.moments(largest)
                   
                    if M["m00"] != 0:
                        black_line_detected = True
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(line, [largest], -1, (0,255,0), 2)
                        cv2.circle(line, (cx, cy), 7, (0,0,255), -1)
    
                        error = cx - line_center_x
                        adjustment = 0.25 * error
                        left_speed  = base_speed - adjustment
                        right_speed = base_speed + adjustment
                        
                        GPIO.output(IN1, GPIO.LOW)
                        GPIO.output(IN2, GPIO.HIGH)
                        GPIO.output(IN3, GPIO.LOW)
                        GPIO.output(IN4, GPIO.HIGH)
                        
                        pwmA.ChangeDutyCycle(max(0,min(100,left_speed)))
                        pwmB.ChangeDutyCycle(max(0,min(100,right_speed)))
            
            if not symbol_pause:
                symbol_input = cv2.resize(symbol, (model_width, model_height))
                input_data = np.expand_dims(symbol_input, axis=0).astype(np.float32) / 255.0

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                predicted_index = np.argmax(output_data)
                confidence = output_data[predicted_index]
                
                if confidence < 0.65:
                    predicted_label = "no object"
                else:
                    predicted_label = labels[predicted_index]

                if confidence < 0.97 and predicted_label == "11 Triangle":
                    predicted_label = "no object"
                    
                cv2.putText(symbol, f'{predicted_label}: {confidence:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if predicted_label not in ["no object", "12 Line"]:
                    symbol_pause = True
                    pause_start_time = time.time()
                    print(f"Symbol Detected: {predicted_label} (Confidence: {confidence:.2f})")
                    pwmA.ChangeDutyCycle(0)
                    pwmB.ChangeDutyCycle(0)

            if symbol_pause:
                if time.time() - pause_start_time < pause_duration:
                    cv2.putText(line, 'PAUSED', (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    symbol_pause = False
                    pwmA.ChangeDutyCycle(base_speed)
                    pwmB.ChangeDutyCycle(base_speed)
                    GPIO.output(IN1, GPIO.LOW)
                    GPIO.output(IN2, GPIO.HIGH)
                    GPIO.output(IN3, GPIO.LOW)
                    GPIO.output(IN4, GPIO.HIGH)
                    time.sleep(0.5)

        if detected_color is not None:
            error = cx_color - line_center_x
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

        cv2.imshow("Line Following View", line)
        cv2.imshow("Symbol Detecting View", symbol)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
