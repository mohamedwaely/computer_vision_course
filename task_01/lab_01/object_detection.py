import cv2
import os
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

if not os.path.exists(r"task_01\lab_01\cars.jpeg"):
    print("Error: Image file not found.")
    exit()

img = cv2.imread(r"task_01\lab_01\cars.jpeg")
img_02 = cv2.imread(r"task_01\lab_01\img02.jpeg")

if img is not None:
    res=model(source=img_02, show=True, conf=0.2, save=True)
else:
    print("Error: Could not read image file.")




