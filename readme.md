## Facial feature extractor

#### Installation

```
CONDA_SUBDIR=osx-64 conda create -n arcface-mxnet python=3.9
pip install mxnet
pip install opencv
pip install scikit-image
pip install numpy
```

#### Model

Download models from [here](https://drive.google.com/drive/folders/1JKqxuIiqKMcjphTKBwHbuJY5toqVu71s) and place them into `model` folder

#### Project structure

Organise the project directory as follows:

``` Shell
├── model
│   ├── MobileFaceNet
│   └── MS1MV2-ResNet100-Arcface
├── resources
│   └── landmarks.csv
└── images
    └── ijbb
        └── ...
```

#### Extraction

To exctract feature vectors use the follwing script

``` Shell
python extraction.py --dataset=ijbb --landmarks=resources/ijbb_small_landmarks.csv --model=MobileFaceNet
```