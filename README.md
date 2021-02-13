# ComputerVisionRegression
Forecasting with Time-Series to Image Encoding and Convolutional Neural Networks
Within forecasting there's an age old question, how useful and how relevant is my data and can I use it to make accurate predictions of what's going to happen in the future. For most of the previous and current century, this has been pretty constant. 
This however has been radically chaining thanks to the introduction of new technologies as well as Deep Learning. 

A novel approach proposed by a team from the University of Cagliari is transforming time-series into images and using Convolutional Neural Networks to find pattern that might otherwise be overlooked by even the most seasoned analyst. 

## Instalation
Download the repo into your machine
Install required packages in requirements.txt
```bash
pip install requirements.txt
```
## Setup
Run setup.py, this will create directories where we store our models, images and data.
```bash
python setup.py
```
## Generate images from time-series data
The script will create Gramian Angular Fields, these will be placed on directories that represent the two classes we are trying to predict; Long or Short

```bash
python images_from_data.py
```
## Create Model
The script will create and fit our ensembled model, measure its perfomance and save it.

```bash
python cnn_model.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
