# LSTM based Sentiment Classification using Tensorflow - Amazon Reviews Rating (Dataset)

<!-- ![Build Status] -->

The dataset is from Amazon Review Data (2018) https://nijianmo.github.io/amazon/index.html.

- I here look at Cell Phones and Accessories review dataset for experimentation. 
- I have pre-processed this dataset in Jupyter Notebook, please find the code in the Preprocessing Folder
- The script for pre-processing is present here: https://github.com/immanuvelprathap/LSTM-Sentiment-Analysis-AmazonReviews-Dataset/tree/main/Preprocessing
- The CellPhonesRating.csv is too large so i cannot upload it but please follow the pre-processing steps here: https://github.com/immanuvelprathap/LSTM-Sentiment-Analysis-AmazonReviews-Dataset/tree/main/PreprocessingDataset/tree/main/Preprocessing

## Directory Tree

```
├── Data
│   ├── CellPhonesRating.csv
│   └── tokenizer.json
├── Images
│   ├── Architecture_LSTM_Model.png
│   ├── The_LSTM_cell.png
│   └── multi_input_and_output_model.png
├── Model
│   ├── LSTM_Model.ipynb
│   └── tf_LSTM_model.h5
├── Preprocessing
│   └── Preprocessing - AmazonReviews Dataset.ipynb
├── README.md
└── gitattributes
```


## Installation - Dependencies

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras import layers
from tensorflow.keras import losses

from collections import Counter


import pandas as pd
import numpy as np

import sklearn


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import seaborn as sns

import pydot
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

<img target="_blank" src="https://miro.medium.com/max/1400/1*-QTg-_71YF0SVshMEaKZ_g.png" width=200>

<img target="_blank" src="https://keras.io/img/logo.png" width=200>

<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width=200>


## Team

[![Immanuvel Prathap]<img target="_blank" src="https://avatars.githubusercontent.com/u/68032323?v=4" width=200>](https://immanuvelprathap.in/)|
-|
[Immanuvel Prathap's Website - Click Here!](https://immanuvelprathap.in/) |)

## License

## Credits

## References for researcher

 
