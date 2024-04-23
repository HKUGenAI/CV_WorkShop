import cv2
import os
import torch
import numpy as np
from groundingdino.util.inference import predict, annotate
from PIL import Image

from groundingdino.util.inference import load_model
import groundingdino.datasets.transforms as T

def transform_image(rgb_image):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(rgb_image, None)
    return image_transformed



projectdir = os.getcwd()
# Define paths
groundingdino_dir = os.path.join(projectdir, "GroundingDINO")
model_config_path = os.path.join(groundingdino_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
weights_path = os.path.join(projectdir, "weights/groundingdino_swint_ogc.pth")

# Load model
model = load_model(model_config_path, weights_path)

# Define constants and paths
TEXT_PROMPT = "cap"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.20


# Initialize webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture image from webcam")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (800, 800))  # Resize to match model input size
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image_tensor = transform_image(image)
    print(image_tensor.shape)

    # Perform object detection
    with torch.no_grad():
        # Perform object detection using Grounding Dino
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

    # Annotate the image
    annotated_frame = annotate(image_source=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), boxes=boxes, logits=logits, phrases=phrases)

    # Display the annotated image
    cv2.imshow("Result", annotated_frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
