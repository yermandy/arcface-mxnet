## Facial feature extractor

#### Installation

```
CONDA_SUBDIR=osx-64 conda create -n arcface-mxnet python=3.9
pip install mxnet
pip install opencv
pip install scikit-image
pip install numpy
```


Organise the project directory as follows:

``` Shell
├── resources
│   └── landmarks.csv
└── images
    └── ijbb
        └── ...
```

#### Extraction

To exctract feature vectors use the follwing script

``` Shell
python extraction.py --dataset=ijbb --landmarks=resources/ijbb_small_landmarks.csv --model=MS1MV2-ResNet100-Arcface
```