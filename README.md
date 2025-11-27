# Real_time-Object-Detection---using-YOLO

# Introduction
This project implements real-time object detection using a webcam feed.
It uses a custom-trained YOLOv8 model integrated with OpenCV to detect objects live on screen.

# How to run this code?
step1:Clone the repository
https://github.com/Nisarga-arun/Real_time-Object-Detection---using-YOLO.git
step2:Install all the necessary libraries
pip install ultralytics
brew install opencv 
pip insall cv2
pip install math
step3:Test and validate a images using roboflow.
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="NcQTPGpmTwPjOgeIqNDn")
project = rf.workspace("nisarga-0lhwb").project("coins-wqct4")
version = project.version(4)
dataset = version.download("yolov8")
step4:Train the model on Google colab.
!nvidia-smi        # check GPU
!pip install ultralytics


from ultralytics import YOLO
import os


!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="NcQTPGpmTwPjOgeIqNDn")
project = rf.workspace("nisarga-0lhwb").project("coins-wqct4")
version = project.version(4)
dataset = version.download("yolov8")


model = YOLO("yolov8n.pt")   # You can use 'yolov8s.pt' or 'yolov8m.pt' for higher accuracy

model.train(
    data="/content/coins-4/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="custom_yolov8_model",
    device=0
)

from google.colab import files
uploaded = files.upload()

# Load your uploaded model
model = YOLO("best.pt")

# If you want in tf.lite format--->Export to TensorFlow Lite
model.export(format="tflite")



from google.colab import files
files.download("best_saved_model/best_float32.tflite")
