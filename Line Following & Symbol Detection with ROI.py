import time
import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter

# ----- GPIO Motor Setup (BOARD numbering) -----
IN1, IN2 = 11, 13
IN3, IN4 = 15, 12
ENA, ENB = 16, 18

GPIO.setmode(GPIO.BOARD)
for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    GPIO.setup(pin, GPIO.OUT)

pwmA = GPIO.PWM(ENA, 100)
pwmB = GPIO.PWM(ENB, 100)
base_speed = 40
pwmA.start(base_speed)
pwmB.start(base_speed)

# ----- TFLite Model Setup -----
MODEL_PATH = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f]

# Initialize interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_h = input_details[0]['shape'][1]
model_w = input_details[0]['shape'][2]

# ----- Motor Control Function -----
def set_motor(left_speed, right_speed):
    # left_speed/right_speed: -100 to 100
    # Left motor
    if left_speed >= 0:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    pwmA.ChangeDutyCycle(min(abs(left_speed), 100))
    # Right motor
    if right_speed >= 0:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    pwmB.ChangeDutyCycle(min(abs(right_speed), 100))

# ----- Symbol Prediction -----
def predict_symbol(roi):
    img = cv2.resize(roi, (model_w, model_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(output)
    return labels[idx], output[idx]

# ----- Main Loop -----
def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={'size': (640, 480)})
    picam2.configure(config)
    picam2.start()

    symbol_pause = False
    pause_start = 0
    ignore_until = 0

    try:
        while True:
            now = time.time()
            frame = picam2.capture_array()
            h, w = frame.shape[:2]

            # Split ROIs
            roi_symbol = frame[0:h//2, :]
            roi_line = frame[h//2:h, :]

            # ---- Line Following ----
            gray = cv2.cvtColor(roi_line, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours and not symbol_pause:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    error = cx - w/2
                    kp = 0.1
                    turn = kp * error
                    left_spd = base_speed - turn
                    right_spd = base_speed + turn
                    set_motor(left_spd, right_spd)
                else:
                    set_motor(0, 0)
            else:
                set_motor(0, 0)

            # ---- Symbol Detection ----
            if not symbol_pause and now > ignore_until:
                symbol, conf = predict_symbol(roi_symbol)
                if conf > 0.65 and symbol.lower() not in ['no object', 'line']:
                    print(f"Detected: {symbol} ({conf:.2f})")
                    symbol_pause = True
                    pause_start = now
                    set_motor(0, 0)

            # Resume logic
            if symbol_pause and now - pause_start >= 2:
                symbol_pause = False
                ignore_until = now + 3

            # ---- Visualization ----
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.line(display, (0, h//2), (w, h//2), (255,0,0), 2)
            cv2.imshow('Robot View', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        pwmA.stop()
        pwmB.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
