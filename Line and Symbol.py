import cv2
import numpy as np
import time
import math
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# ----- GPIO Setup -----
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
base_speed = 20
pwmA.start(base_speed)  
pwmB.start(base_speed)

# ----- Load TFLite Model and Labels -----
MODEL_PATH = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'

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

# ----- Camera Initialization -----
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# ----- Control Flags and Timing -----
symbol_pause = False
pause_duration = 3
pause_start_time = 0
line_center_x = 320  # Center of the frame for line following

try:
    while True:
        # --- Capture a fresh frame from the camera ---
        raw_frame = picam2.capture_array()
        if raw_frame.shape[2] == 4:
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2RGB)

        # --- Create two independent copies ---
        frame_line = raw_frame.copy()
        frame_symbol = raw_frame.copy()

        # --- Line Following Processing (Always active) ---
        # Convert to grayscale (CORRECTED COLOR SPACE)
        gray = cv2.cvtColor(frame_line, cv2.COLOR_RGB2GRAY)
        
        # Adjust these values based on your lighting conditions
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)  # Lowered threshold
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on line following view
        cv2.drawContours(frame_line, contours, -1, (0, 255, 0), 2)

        # --- Symbol Detection ---
        if not symbol_pause:
            # Symbol detection processing
            symbol_input = cv2.resize(frame_symbol, (model_width, model_height))
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

            # Draw detection info
            cv2.putText(frame_symbol, f'{predicted_label}: {confidence:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Handle symbol detection
            if predicted_label not in ["no object", "12 Line"]:
                symbol_pause = True
                pause_start_time = time.time()
                print(f"Symbol Detected: {predicted_label} (Confidence: {confidence:.2f})")
                pwmA.ChangeDutyCycle(0)
                pwmB.ChangeDutyCycle(0)

        # --- Line Following Control ---
        if contours and not symbol_pause:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - line_center_x
                adjustment = 0.25 * error
                left_speed = max(0, min(100, base_speed - adjustment))
                right_speed = max(0, min(100, base_speed + adjustment))

                pwmA.ChangeDutyCycle(left_speed)
                pwmB.ChangeDutyCycle(right_speed)

                # Draw center line and contour center
                #cv2.line(frame_line, (line_center_x, 0), (line_center_x, 480), (255, 0, 0), 2)
                cv2.circle(frame_line, (cx, int(M["m01"]/M["m00"])), 7, (0, 0, 255), -1)

                # Forward motion
                GPIO.output(IN1, GPIO.LOW)
                GPIO.output(IN2, GPIO.HIGH)
                GPIO.output(IN3, GPIO.LOW)
                GPIO.output(IN4, GPIO.HIGH)
                
        elif not symbol_pause:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)

        # --- Pause Handling ---
        if symbol_pause:
            if time.time() - pause_start_time < pause_duration:
                cv2.putText(frame_line, 'PAUSED', (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                symbol_pause = False
                time.sleep(0.5)

        # --- Display Windows ---
        cv2.imshow("Line Following View", frame_line)
        cv2.imshow("Symbol Detection View", frame_symbol)
        cv2.imshow("Threshold", thresh)  # Show threshold view

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
