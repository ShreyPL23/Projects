import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

### Load the MiDaS model
# The MiDaS model is a deep learning model for monocular depth estimation.
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to ("cuda")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

cap = cv2.videoCapture(0)  # Open the webcam
while cap.isOpend():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    cv2.imshow("Webcam", frame)  # Display the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    