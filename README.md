# Deep-Learning-Lunge-Detection

This repository contains a deep learning method to automatically label whale lunges from a time series of accelerometer data. The problem input is a finite time series of data. The problem output is a finite set of lunge times, indicated by defining features in the time series. 

To label the lunges, we use a sliding window approach. We train a labeling model that inputs a fixed length window (say a 20 second window) of the time series. The network predicts whether there a lunge within some fixed time (say 2 seconds) of the middle of the window. To obtain the lunge predictions, we slide the fixed-length window across the entire time series, obtain the neural network's predictions on these windows, and consolidate the predictions to produce the set of lunge times. Finally, we use a correction model to place the predictions closer to the true lunge times.

## Installation

First clone the project

```
git clone https://github.com/valdivia4/Deep-Learning-Lunge-Detection.git
cd Deep-Learning-Lunge-Detection
```

To install the Python dependencies, we recommend using a virtual environment along with either anaconda or pip.  Use the following commands to make and activate the virtual environment.

```
source venv/bin/activate
virtualenv venv
```

Then install the required packages from requirements.txt

```
pip3 install -r requirements.txt
```

## Example Usage

We include synthetic data with the project and recommend to run the code first with this synthetic data. The steps for using your own data will be mostly identical. Here we run through example usage using the synthetic data.

### Preprocessing
For training, the first step is to preprocess the data, which happens in the preprocessing directory. 

```
cd preprocessing
```

The raw lunge files for training, validation, and testing go in the raw_data directory. We assume a .csv input where raw_data/inputs is the time series of measurements and raw_data/labels is the true labels of the lunges. For specifics on formatting, see the synthetic data provided in these folders.

The next step is to convert the data to numpy format

```
python3 convert_to_numpy.py
```

which cleans and normalizes the data and saves it .npy format in the created numpy_data directory.

Next, modify the variables in the data_config.py file to conform to your data. This file contains information about the train/dev splits, information about the deployments such as the sampling frequency, and information about the desired padded_window and window sizes for training.

Then we create the windows used for training the labeling model and the correction model.

```
python3 generate_train_windows
python3 generate_correction_model_windows
```




