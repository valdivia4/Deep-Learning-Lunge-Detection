import os
import sys
sys.path.insert(0, '../utils')
from utils import *
sys.path.insert(0, '../preprocessing')
from convert_to_numpy import process_csv
sys.path.insert(0, '../train')
from model_configs import get_weighted_bce
from keras.utils.generic_utils import get_custom_objects
import keras
import keras.backend as K

#defining custom loss function
pos_weight = 1 #this setting doesn't matter here, just helps to load the model
weighted_bce = get_weighted_bce(pos_weight)
get_custom_objects().update({"weighted_bce": weighted_bce})

#load label model
#SET THESE: the folder and the name of the desired label model, as well as flattened_input
folder = 'feed_forward_Tue_Oct__1_16-54-25_2019'
model_name = 'ep_1_tp_0.966_fp_0.0_f_1_0.982_f_2_0.972_chain_2_thresh_0.5'
flattened_input = True #true for feed forward, false for resnet

model_name_split = model_name.split('_')
thresholds = [float(model_name_split[-1])]
chaining_dists = [float(model_name_split[-3])]
model = keras.models.load_model('../models/label_models/' + folder + '/' + model_name,
                               custom_objects={'loss': weighted_bce })

def avgabs(y_true,y_pred): ##in seconds (if perturbation_max = 5*fs)
    return K.mean(K.abs(5*(y_true - y_pred)))

# SET THIS: choose which correction model to use by uncommenting the appropriate
# line below

corr_model_type = 'classification'
#corr_model_type = 'regression'
#corr_model_type = None

if corr_model_type == 'classification':
    correction_model = keras.models.load_model('../models/correction_models/correction_model_class.h5')
elif corr_model_type == 'regression':
    correction_model = keras.models.load_model('../models/correction_models/correction_model_reg.h5', custom_objects={'avgabs': avgabs})
elif corr_model_type is None:
    correction_model = None

# now generate the labels
raw_path = './unlabeled_inputs/'
for input_filename in os.listdir(raw_path):
    np_features = process_csv(raw_path, input_filename)

    labels, __ = get_predictions(np_features, model, flattened_input,
                                 correction_model, corr_model_type)
    labels = np.array(labels)
    labels = np.reshape(labels, (len(labels),1))

    label_path = 'predicted_labels'
    if not os.path.exists(label_path):
       os.makedirs(label_path)
    np.savetxt(label_path + '/labels_{}'.format(input_filename), labels, delimiter=',')
