# If you want to train your own model using your own dataset, you can use Colab notebook.

!pip install ultralytics
!pip install roboflow

# rugby league dataset
from roboflow import Roboflow
rf = Roboflow(api_key="y80duBYM8z5s5hJy35Bi")
project = rf.workspace("aap-blocky-yqzrb").project("nrl-player-detection")
version = project.version(2)
dataset = version.download("yolov8")

# Training
!yolo task=detect mode=train model='yolov8s.pt' data={dataset.location}/data.yaml epochs=100 imgsz=640 patience=10
