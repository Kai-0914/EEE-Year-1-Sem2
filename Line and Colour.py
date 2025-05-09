import RPi.GPIO as GPIO
from picamera2 import Picamera2
import cv2
import time
import numpy as np

# Motor control pins
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

# Color definitions (HSV ranges)
colors = [
    {   # Blue
        "lower": np.array([90, 50, 50]),
        "upper": np.array([130, 255, 255]),
        "name": "Blue",
        "color": (0, 0, 255)
    },
    {   # Green
        "lower": np.array([40, 50, 50]),
        "upper": np.array([80, 255, 255]),
        "name": "Green",
        "color": (0, 255, 0)
    },
    {   # Red (dual range)
        "lower": [np.array([0, 50, 50]), np.array([160, 50, 50])],
        "upper": [np.array([10, 255, 255]), np.array([180, 255, 255])],
        "name": "Red",
        "color": (255, 0, 0)
    },
    {   # Yellow
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255]),
        "name": "Yellow",
        "color": (139, 128, 0)
    }
]

# Specify your priority colors here
PRIORITY_COLORS = ['Red', 'Blue']  # Change these to your desired colors

def get_color_info(color_name):
    for color in colors:
        if color['name'] == color_name:
            return color
    return None

priority_colors = [get_color_info(name) for name in PRIORITY_COLORS]

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    image_center_x = 640 // 2

    try:
        while True:
            frame = picam2.capture_array()
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # Reset flags
            target_cx = None
            detected_color = None
            black_cx = None
            black_cy = None

            # 1) Look for priority colors exactly as before…
            for color in priority_colors:
                if color is None:
                    continue

                if color['name'] == 'Red':
                    mask1 = cv2.inRange(hsv, color['lower'][0], color['upper'][0])
                    mask2 = cv2.inRange(hsv, color['lower'][1], color['upper'][1])
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, color['lower'], color['upper'])

                mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
                mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) < 500:
                    continue
                M = cv2.moments(largest)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                target_cx = cx
                detected_color = color
                # draw…
                cv2.drawContours(frame, [largest], -1, color['color'], 2)
                cv2.circle(frame, (cx, cy), 7, color['color'], -1)
                cv2.putText(frame, f"{color['name']} ({cx},{cy})",
                            (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color['color'], 2)
                break

            # 2) If no priority color, look for black line
            if detected_color is None:
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, 60])
                black_mask = cv2.inRange(hsv, lower_black, upper_black)
                black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)), iterations=2)
                black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN,
                                              cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)), iterations=2)
                contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M['m00'] != 0:
                        black_cx = int(M['m10']/M['m00'])
                        black_cy = int(M['m01']/M['m00'])
                        # draw…
                        cv2.drawContours(frame, [largest], -1, (0,255,0), 3)
                        cv2.circle(frame, (black_cx, black_cy), 10, (0,0,255), -1)

            # 3) Decide motion
            if detected_color is not None:
                # steer on colored line
                error = target_cx - image_center_x
                adjustment = 0.25 * error
                left_speed  = base_speed - adjustment
                right_speed = base_speed + adjustment

                # set forward direction
                GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
                GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
                pwmA.ChangeDutyCycle(max(0,min(100,left_speed)))
                pwmB.ChangeDutyCycle(max(0,min(100,right_speed)))

            elif black_cx is not None:
                # steer on black line
                error = black_cx - image_center_x
                # stronger correction at bottom
                if 400 > black_cy > 300:
                    adjustment = 2.00 * error
                elif black_cy >= 400:
                    pwmA.ChangeDutyCycle(70)
                    pwmB.ChangeDutyCycle(70)
                    if black_cx < 320:
                        GPIO.output(IN1, GPIO.LOW)
                        GPIO.output(IN2, GPIO.HIGH)
                        GPIO.output(IN3, GPIO.HIGH)
                        GPIO.output(IN4, GPIO.LOW)
                    else:
                        GPIO.output(IN1, GPIO.HIGH)
                        GPIO.output(IN2, GPIO.LOW)
                        GPIO.output(IN3, GPIO.LOW)
                        GPIO.output(IN4, GPIO.HIGH)
                else:
                    adjustment = 0.25 * error
                        
                left_speed  = base_speed - adjustment
                right_speed = base_speed + adjustment
                GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
                GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
                pwmA.ChangeDutyCycle(max(0,min(100,left_speed)))
                pwmB.ChangeDutyCycle(max(0,min(100,right_speed)))

            else:
                # NO LINE DETECTED → REVERSE!
                GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
                GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
                pwmA.ChangeDutyCycle(40)
                pwmB.ChangeDutyCycle(40)

            # show frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Line Following", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
