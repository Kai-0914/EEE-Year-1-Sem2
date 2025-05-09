import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# ----- Load the TFLite Model and Labels -----
MODEL_PATH = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'  # Update if you have a labels file

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(LABELS_PATH) if LABELS_PATH else None

# Initialize the TFLite interpreter and allocate tensors
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Retrieve input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract expected input size from the model (height and width)
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# ----- Set Up Picamera2 -----
picam2 = Picamera2()
# Create a preview configuration; adjust the preview size if needed.
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

try:
    while True:
        # Capture a frame using Picamera2
        frame = picam2.capture_array()
        
        # If the captured frame has 4 channels, convert it to 3 channels (RGBA to RGB)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Convert the image from RGB to BGR for display, if needed later
        # For inference, use the original frame (assumed to be in RGB)
        resized_frame = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0
        
        # Run inference with the TensorFlow Lite interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Obtain model predictions; adjust this part if your model outputs differ
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index]
        if confidence < 0.65:
            predicted_label = "no object"
        else:
            predicted_label = labels[predicted_index] if labels else str(predicted_index)
        
        # For display, convert the RGB frame to BGR so that colors appear normal in OpenCV
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        text = f'{predicted_label}: {confidence:.2f} ({inference_time*1000:.1f} ms)'
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Frame", display_frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Clean up: stop the camera and close any open OpenCV windows
    picam2.stop()
    cv2.destroyAllWindows()
