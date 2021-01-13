# Face Dectection using EfficientDet-D0 trained on WIDER_FACE dataset

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

This Project utilises the TensorFlow Object Detection API.

Clone the [repo](https://github.com/sumitIO/FaceDectection-using-EfficientDet.git) using:
```
git clone https://github.com/sumitIO/FaceDectection-using-EfficientDet.git
```
Or use the IPYTHON file [this](https://github.com/sumitIO/FaceDectection-using-EfficientDet/blob/main/EfficientDet_Model_Training.ipynb) to reproduce the code.

Note : TF Object Detection API must be install!!!
See the offical Page for API Installation [Here](https://github.com/tensorflow/models/tree/master/research/object_detection)

# Steps to Train/Evalute Model Locally

## 1. Clone the TensorFlow Oject Detection API
```
git clone https://github.com/tensorflow/models.git

<!-- Install the Object Detection API -->
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

## 2. Download the Wider Face Train Data & Validation Data
```
bash downloadData.sh
```
## 3. Preprocess Data
```
bash preprocessData.sh
```
## 4. Generate TFRecord File
```
bash generate_tfrecord.py
```
## 5. Split training TFRecord file
```
bash create_shrads.sh
```
## 6. Training Model
```

```

## 7. Evaluate Model
```

```
## 6. Detect Faces
Before start making detections make sure you have replace the image fie with your own image in /Utility/DetectFace.py.
```
bash detect_face.sh
```


# Steps to Train/Evalute in Colab

To reproduce the code step-by-step:

run Google Colab NoteBook from [Here](https://github.com/sumitIO/FaceDectection-using-EfficientDet/blob/main/EfficientDet_Model_Training.ipynb)