import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your YOLOv26 model (trained for butterfly detection)
yolo_model = YOLO('butterflies_YOLOv26n.pt')  # Replace with your YOLO model path

# Load your EfficientNet classification model
classification_model = tf.keras.models.load_model('best_butterfly_EfficientNetV2B2.keras')  # Replace with your model path

# Load species names from categories.txt
with open('categories.txt', 'r') as f:
    content = f.read()
    # Extract the list inside CATEGORIES = [...] format
    species_names = eval(content.split('=')[1].strip())

# Define the function to perform detection and classification
def detect_and_classify(image, conf_threshold, iou_threshold, max_det):
    # Convert the uploaded image from PIL to a NumPy array (OpenCV format)
    image_np = np.array(image)

    # Run YOLOv26 inference with max detections, confidence threshold, and IoU threshold
    results = yolo_model(image_np, conf=conf_threshold, iou=iou_threshold, max_det=max_det)

    # Extract the bounding boxes from YOLO results
    boxes = results[0].boxes  # Get boxes from the first result (assuming one image)

    # Initialize a variable to store classification results
    classification_output = f"Number of butterflies detected: {len(boxes)}\n\n"
    
    if len(boxes) == 0:
        return image, "No butterflies detected"

    # Draw bounding boxes and classify each detected butterfly
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        
        # Crop the image to the detected butterfly
        cropped_butterfly = image_np[y1:y2, x1:x2]
        
        # Resize the cropped image to 224x224 for the EfficientNet model
        cropped_butterfly_resized = cv2.resize(cropped_butterfly, (224, 224))
        cropped_butterfly_resized = np.expand_dims(cropped_butterfly_resized, axis=0)  # Add batch dimension
        
        # Run the EfficientNet classification model
        predictions = classification_model.predict(cropped_butterfly_resized)[0]
        
        # Get the top 3 species and their confidence scores
        top_3_indices = predictions.argsort()[-3:][::-1]
        top_3_species = [(species_names[i], predictions[i]) for i in top_3_indices]
        
        # Add classification results for this butterfly
        classification_output += f"Butterfly {i+1}:\n" + "\n".join([f"{species}: {score:.2f}" for species, score in top_3_species]) + "\n\n"
        
        # Draw the bounding box and classification results on the image
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the box in red
        confidence_text = f"{box.conf[0]:.2f}"  # Confidence score of the YOLO detection
        cv2.putText(image_np, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Add top 1 species name near the bounding box
        top_species_name = top_3_species[0][0]
        cv2.putText(image_np, top_species_name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert the NumPy image back to PIL format for Gradio display
    output_image = Image.fromarray(image_np)

    return output_image, classification_output

# Set up the Gradio interface
gr.Interface(
    fn=detect_and_classify,  # Function that performs both detection and classification
    inputs=[
        gr.Image(type="pil"),  # Accept an uploaded image
        gr.Slider(minimum=0.01, maximum=1.0, value=0.25, label="Confidence Threshold"),  # Slider for confidence threshold
        gr.Slider(minimum=0.01, maximum=1.0, value=0.45, label="IoU Threshold"),  # Slider for IoU threshold
        gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Max Detections")  # Slider for max detections
    ],
    outputs=[
        gr.Image(),  # Output will be the image with bounding boxes
        gr.Textbox(label="Classification Results")  # Display classification results
    ],
    title="Digital Entomology Web App ",
    description="Upload an image, and the YOLOv26 model will detect the butterflies. The EfficientNet model will classify the species of each detected butterfly."
).launch()
