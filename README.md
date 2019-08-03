# Deep-Learning-Lunge-Detection

This repository contains an implementation for automatically labeling whale lunges from a time series of accelerometer data. The problem input is a finite time series of data. The problem output is a finite set of lunge times, indicated by defining features in the time series. 

To label the lunges, we use a sliding window approach. We train a labeling model that inputs a fixed length window (e.g. a 20 second window) of the time series. The network predicts whether there a lunge within some fixed time (e.g. 2 seconds) of the middle of the window. To obtain the lunge predictions, we slide the fixed-length window across the entire time series, obtain the neural network's predictions on these windows, and consolidate the predictions to produce the set of lunge times. Finally, we use a correction model to place the predictions closer to the true lunge times.

We have tried to make this tutorial and project usable for people (particularly in the biology research community) with minimal prior python experience. 

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

The raw lunge files for training, validation, and testing go in the raw_data directory. We assume a .csv input where raw_data/inputs contains the time series of measurements and raw_data/labels contains the true labels of the lunges. For specifics on formatting, see the synthetic data provided in these folders.

The next step is to convert the data to numpy format

```
python3 convert_to_numpy.py
```

which performs some preprocessing on the data and saves it .npy format in the created numpy_data directory.

Next, modify the variables in the data_config.py file to conform to your provided data. 

These variables contain information about the train/dev splits, information about the deployments such as the sampling frequency, and information about the desired padded_window and window sizes for training. Information about the variable meanings are commented in the file.

Then we create the windows used for training the labeling model.

```
python3 generate_train_windows
```

These windows are saved in the newly created Deep-Learning-Lunge-Detection/training_windows directory. 

We can also create the windows for the correction model. Do this step after training the labeling model and evaluating it (see next two sections), as the hyperparameters for the correction model depend on the labeling model's performance.
 
```
python3 generate_correction_model_windows
```

### Training

The next step is to train a labeling model. To do so, move to the Deep-Learning-Lunge-Detection/train directory.

There are two classes of models we currently support: a feed forward network and a 1D-ResNet. To design a models, edit the the appropriate variables in the model_configs file. (See the file comments for information on these variables.)

Next, set the config_name in the appropriate file. We will use the feed forward network here with the default feed_forward_config settings. So we would set config_name='feed_forward' in feed_forward_model.py. To train the model, we call

```
python3 feed_forward_model.py
```

And the model should begin training. After each epoch, the validation set metrics are printed and the current model (named by its validation set metrics) and the model config values are saved in the newly created Deep-Learning-Lunge-Detection/models/label_models/*experiment_date_and_time* directory.
 
 To train the ResNet, follow the same steps using the resnet_model.py file instead of feed_forward_model.py.
 
 Finally, to train the correction model, call
 
 ```
 python3 correction_model_regression.py
 ```
 
 ### Evaluation
 
 Evaluation of model performance takes place in the Deep-Learning-Lunge-Detection/evaluate directory. This directory contains two Jupyter notebooks for evaluation. To start the Jupyter, call
 
 ```
 jupyter notebook
 ```
 
 in the evaluate directory.
 
 #### Plotting
 The first notebook plot_lunge_predictions plots a model's predictions on a deployment. To set which model to use, set the folder and model name, e.g.
 ```
 folder = 'feed_forward_Sat_Aug__3_11-51-22_2019'
 model_name = 'ep_2_tp_0.983_fp_0.0_f_1_0.991_f_2_0.986_chain_2_thresh_0.5'
 ```
 
 Then run the cells. You can select which range to plot the true labels and the model predictions by setting the startTime and endTime variables.
 
 Here is an example output on the synthetic data. The dots correspond to the true labels, and the triangles correspond to model predictions.
 
 ![alt text](readme_images/sample_plot_output.png "Sample plot output")
 
 #### Model Metrics
 
 The second notebook computes various model metrics on a deployment
 
 
 
 

