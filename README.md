# Image-based traffic analysis and predictor.
## Description
This project is designed to implement two use-cases of Artificial Intelligence machine-learning: Time-
Series data analysis and Image Analysis. In this case, the image-analysis component is designed to
used an existing object-recognition model built on the YOLOV8 model to ”look” at an intersection,
using Keras-CV, to detect vehicular traffic movement over time at a single intersection. The data is
then used for a machine-learning algorithm which can then be used to predict the traffic flow based
on the day and time.

The specific intersection is located at Coldwater, Michigan and the local township has provided
a live video stream at https://www.youtube.com/watch?v=ByED80IKdIU. The live-stream is down-
loaded constantly, with downloaded videos sorted per day.

To implement the data analysis model model, you must first understand that this is a linear regression machine learning model that predicts total amount of cars crossing the intersection in a time-series manner.
This program compares two different models produced from the time:
```
Time vs Total.
```
And
```
Time, N, W, E, S vs Total
```

We are taking 80% of the data to train the model and 20% to test the model.

Interestingly enough, the latter model was spectacularly more accurate according to r2 score.
## Installation and Setup
The installation and setup processes are fairly straight-forward.
To install and use the Image Analysis component, several dependencies must be installed first.
1. First, you must ensure you are using Debian 12.8. Debian 12.8 is the only tested operating
system version that this component has been tested on.
2. Before continuing, please run the following command to install important, initial packages:
```
sudo apt-get install software-properties-common python3-venv
```
3. After installation, if you have a NVIDIA CUDA capable graphics card, you must install CUDA
and nvidia-drivers. Guides may be found at https://developer.nvidia.com/cuda-12-4-0-download-archive
for CUDA and https://wiki.debian.org/NvidiaGraphicsDrivers for the nvidia-drivers.
4. Next, create a directory/folder and place the ”imageAnalyzer.py” file in it.
5. Afterwards, create a new python virtual environment, after entering that directory, using
```
pythom3 -m venv env
```
6. Then launch the environment using
```
source env/bin/activate
```
7. Then run
```
source/env/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
To install torch.
8. Afterwards, run
```
source/env/bin/python -m pip install --upgrade keras keras-cv tensorflow-cpu jax jaxlib matplotlib opencv-python
```
9. From there, you are able to now use
```
source/env/bin/python -m imageAnalyzer.py
```
To start the program. You’ll be walked through the steps to analyze a video! A sample video,
sample.mp4 and sample.txt, has been provided. You will need to know the video frames per
second to use the program, and ensure the video file is in the same folder as the .py file.
NOTE: This program has been designed only for the above-mentioned intersection!

For the traffic prediction, you must either use Google Collab, or Jupyter notebook.
The environment must have the following packages installed:
```
sklearn
matplotlib
pandas
```

## Developer Information
It is possible to change the "hit regions" of the detector by changing the values in the
```
checkIfInRegion(centerPointX, centerPointY)
```
To fine-tune to a new environment.

Furthermore, it is possible to change the model in use by modifying
```
pretrained_model
```

Otherwise, much of the system is automated, allowing for different frame-rates, videos and lengths.
