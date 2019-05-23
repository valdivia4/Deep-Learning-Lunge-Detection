import os
import sys
sys.path.insert(0, '../utils')
from utils import *
sys.path.insert(0, '../preprocessing')
from convert_to_numpy import process_csv
sys.path.insert(0, '../train')
from model_configs import get_weighted_bce
from keras.utils.generic_utils import get_custom_objects

#first load the model
#defining custom loss function
pos_weight = 8 #if using weighted binary crossentropy loss
weighted_bce = get_weighted_bce(pos_weight)
get_custom_objects().update({"weighted_bce": weighted_bce})

#load label model
folder = 'feed_forward_Wed_May_22_10-55-38_2019'
model_name = 'ep_1_tp_0.977_fp_0.107_f_1_0.933_f_2_0.959_chain_4_thresh_0.9'

flattened_input = True #change this depending on the model

model_name_split = model_name.split('_')
thresholds = [float(model_name_split[-1])]
chaining_dists = [float(model_name_split[-3])]
model = keras.models.load_model('../models/label_models/' + folder + '/' + model_name,
                               custom_objects={'loss': weighted_bce })

def avgabs(y_true,y_pred): ##in seconds (if perturbation_max = 5*fs)
    return K.mean(K.abs(5*(y_true - y_pred)))

correction_model = keras.models.load_model('../models/correction_models/correction_model.h5', custom_objects={'avgabs': avgabs})
#correction_model = None

#now generate the labels
raw_path = './unlabeled_inputs/'
for input_filename in os.listdir(raw_path):
    np_features = process_csv(raw_path, input_filename)


    labels, __ = get_predictions(np_features, model, flattened_input, correction_model)
    labels = np.array(labels)
    labels = np.reshape(labels, (len(labels),1))

    label_path = 'predicted_labels'
    if not os.path.exists(label_path):
       os.makedirs(label_path)
    np.savetxt(label_path + '/labels_{}'.format(input_filename), labels, delimiter=',')
