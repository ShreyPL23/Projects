import cv2
import torch
import numpy as np
import torch.nn.functional as F

## To run this code call the script with the following command:
#python test.py
#Left-click in the Webcam window on a point whose real distance you know.
#Type the prompted distance (in meters) into the terminal and press Enter.
#Left-click on any other point in the Webcam window to measure its distance.
#Press 'r' to reset the calibration.
#Press 'q' to quit.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform  = transforms.small_transform

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

calibration_scale = None
display_click_dist = None
click_pt = None

def prompt_distance():
    """ Keep asking until the user types a valid float. """
    while True:
        s = input("Known distance at this point (m)? ").strip()
        try:
            return float(s)
        except ValueError:
            print(" Invalid number. Please type something like 2.3 and press Enter.")

def mouse_callback(event, x, y, flags, param):
    global calibration_scale, display_click_dist, click_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        z = depth_map[y, x]
        if calibration_scale is None:
            # first click → get a valid float
            D0 = prompt_distance()
            calibration_scale = D0 / z
            print(f"[Calibration] scale = {calibration_scale:.4f} m per depth unit")
        else:
            # subsequent clicks → just measure
            display_click_dist = z * calibration_scale
            click_pt = (x, y)

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Depth inference (same as before)…
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        if device.type=='cuda':
            with torch.cuda.amp.autocast():
                pred = midas(input_batch)
        else:
            pred = midas(input_batch)

    pred = F.interpolate(
        pred.unsqueeze(1),
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False
    ).squeeze()
    depth_map = pred.cpu().numpy()

    # …visualization and overlay code unchanged…

    cv2.imshow("Webcam", frame)
    cv2.imshow("Depth Map", depth_color)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        calibration_scale = None
        display_click_dist = None
        click_pt = None
        print("[Reset] calibration")

cap.release()
cv2.destroyAllWindows()
