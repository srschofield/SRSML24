# Define path where to find the module. This allows for a different path depending on where the code is running (my mac or the cluster)
import os

module_path_list = [
    '/Users/steven/academic-iCloud/Python/modules',
    '/hpc/srs/Python/modules'
] 

data_path_list = [
    '/Users/steven/Python-data',
    '/hpc/srs/Python-data'
]

module_path = next((p for p in module_path_list if os.path.exists(p)), None)
if not module_path:
    exit("No valid module paths.")
else:
    print('module_path = {}'.format(module_path))

data_path = next((p for p in data_path_list if os.path.exists(p)), None)
if not module_path:
    exit("No valid data paths.")
else:
    print('data_path = {}'.format(data_path))

# adjust tensorflow output level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 default all messages, 1 warnings and errors, 2, errors, 3 fatal errors only

# Import standard modules
import os, sys
import numpy as np

import platform

from datetime import datetime

# Add custom module path to list
sys.path.append(module_path)

# Import custom module
import SRSML24.data_prep as dp
import SRSML24.model as m

import tensorflow as tf
#from tensorflow.keras.optimizers.legacy import Adam 
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

#import platform 

m.print_system_info()

start_time = dp.current_datetime()

# Parameters for windows creation
job_name = 'all_data_2023_script'
job_data_path = dp.create_new_data_path(data_path, job_name, include_date=False)
mtrx_train_path = os.path.join(data_path, 'mtrx/train')
mtrx_test_path = os.path.join(data_path, 'mtrx/test')
mtrx_predict_path = os.path.join(data_path, 'mtrx/predict')
flatten_method = 'poly_xy'
pixel_density = 15.0    # Convert all images to a constant pixel density
pixel_ratio = 0.7       # If an image has less than this % in the slow scan direction it is discarded
data_scaling = 1.e9     # Scale the z-height of the data
window_size = 32        # Window size for training/validation
window_pitch = 16       # Window pitch for training/validation
together = True        # Set this True to save image windows for a mtrx image as a single file rather than separate files.
collate = False         # Set this True to remove all subfolder directories and save all data in root data path

# Parameters for training
model_name = 'unet_' + job_name
batch_size = 128
buffer_size = 12800 # shuffling
learning_rate = 1e-4
epochs = 5

# Parameters for clustering
cluster_model_name = model_name + '_kmeans'
cluster_batch_size = 5120 # This is the number of latent features in a batch for clustering. 
                          # Does not have to be the same as for training and probably should 
                          # be larger. 
cluster_buffer_size = cluster_batch_size * 5    # shuffling buffer
num_clusters=20                                # Desired number of clusters (centroids) to form in the data.
n_init=50                                       # Number of times the algorithm will run with different centroid seeds.
max_iter=1000                                   # Maximum iterations allowed for each mini-batch to refine centroids.
reassignment_ratio=0.05                         # Fraction of clusters reassigned per step; lower values stabilize updates.


# Parameters for PREDICTIONS
predict_window_pitch = 4               # Window pitch for prediction


# DATA LIMITS FOR TESTING THE CODE
mtrx_train_data_limit = None #500         # Number of MTRX files to process (training)
mtrx_test_data_limit = None #100         # Number of MTRX files to process (validation)

train_data_limit = None #2000*batch_size
predict_data_limit = None

# REMOVE ALL DATA FOLDERS EXCEPT MTRX 
dp.delete_data_folders(job_data_path, override=True)

# Training data
mtrx_train_file_list, _ = dp.list_files_by_extension(mtrx_train_path,'Z_mtrx',verbose=False)

dp.process_mtrx_files(
    mtrx_train_file_list[0:mtrx_train_data_limit],
    job_data_path, # save data path
    flatten_method = flatten_method, pixel_density = pixel_density, pixel_ratio = pixel_ratio,
    data_scaling = data_scaling, window_size = window_size, window_pitch = window_pitch,
    save_windows = True,
    save_jpg = True,
    together = together,
    collate = collate,
    verbose = False
    )

# Test data
mtrx_test_file_list, _ = dp.list_files_by_extension(mtrx_test_path,'Z_mtrx',verbose=False)

dp.process_mtrx_files(
    mtrx_test_file_list[0:mtrx_test_data_limit],
    job_data_path, # save data path
    flatten_method = flatten_method, pixel_density = pixel_density, pixel_ratio = pixel_ratio,
    data_scaling = data_scaling, window_size = window_size, window_pitch = window_pitch,
    save_windows = True,
    save_jpg = True,
    together = together,
    collate = collate,
    verbose = False
    )

