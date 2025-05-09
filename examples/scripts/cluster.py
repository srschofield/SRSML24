#!/usr/bin/env python
# coding: utf-8

# <!-- # CNN autoencoder and Clustering from MTRX data
# 
# Use this notebook to load Scienta Omicron Matrix format SPM data and create standardised images for machine learning training and analysis. The code can generate both JPG image data, useful for manually checking the data, and windowed numpy data that can be loaded into ML models. 
# 
# The notebook then creates an autoencoder for training on a large dataset, followed by KMEANS clustering. 
# 
# **Author**: Steven R. Schofield  
# **Created**: November, 2024 -->


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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 0 default all messages, 1 warnings and errors, 2, errors, 3 fatal errors only

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
job_name = 'data_10000_1000'
job_data_path = dp.create_new_data_path(data_path, job_name, include_date=False)
mtrx_train_path = os.path.join(data_path, 'mtrx/train')
mtrx_test_path = os.path.join(data_path, 'mtrx/test')
mtrx_predict_path = os.path.join(data_path, 'mtrx/predict')
flatten_method = 'poly_xy'
pixel_density = 15.0    # Convert all images to a constant pixel density
pixel_ratio = 0.85       # If an image has less than this % in the slow scan direction it is discarded
data_scaling = 1.e9     # Scale the z-height of the data
window_size = 32        # Window size for training/validation
window_pitch = 32       # Window pitch for training/validation
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
mtrx_train_data_limit = 10000         # Number of MTRX files to process (training)
mtrx_test_data_limit = 1000         # Number of MTRX files to process (validation)

train_data_limit = None 
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

# Training data - tensorflow data pipeline for autoencoder
windows_train_path = os.path.join(job_data_path, 'windows/train')
train_files, num_train = dp.list_files_by_extension(windows_train_path, 'npy')
train_files = train_files[:train_data_limit]

# Create dataset with prefetching
train_dataset = m.create_tf_dataset_batched(
    train_files, 
    batch_size=batch_size, 
    buffer_size=buffer_size, 
    window_size=window_size,
    is_autoencoder=True, 
    shuffle=True)

# Validation data - tensorflow data pipeline for autoencoder
windows_test_path = os.path.join(job_data_path, 'windows/test')
test_files, num_test = dp.list_files_by_extension(windows_test_path, 'npy')
test_files = test_files[:train_data_limit]

# Create dataset with prefetching
test_dataset = m.create_tf_dataset_batched(
    test_files, 
    batch_size=batch_size, 
    buffer_size=buffer_size, 
    window_size=window_size,
    is_autoencoder=True, 
    shuffle=True)

# Build and compile the UNET model
autoencoder_model = m.build_autoencoder(window_size=window_size,model_name=model_name)
autoencoder_model.summary()

# Check if running on Apple Silicon
is_mac_silicon = platform.system() == "Darwin" and platform.processor() == "arm"

if is_mac_silicon:
    print("Running on Mac with Apple Silicon. Using legacy RMSprop optimizer.")
    autoencoder_model.compile(
        optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mse', 'mae']
    )
else:
    print("Using new RMSprop optimizer.")
    autoencoder_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mse', 'mae']
    )


# Train the model using the tf.data datasets
history = autoencoder_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2
)

# Save the model as soon as training completes
model_path = os.path.join(job_data_path,'model')
m.save_model(autoencoder_model, model_path, model_name=model_name)

end_time = dp.current_datetime()
dp.elapsed_time(start_time,end_time)

# Plot training history
m.plot_training_history(history, 
                        loss_name='loss', 
                        val_loss_name='val_loss', 
                        metric_names=['mse', 'mae'], 
                        save_to_disk=True,
                        val_metric_names=['val_mse', 'val_mae'],
                        output_path=model_path,
                        dpi=150)


dp.delete_data_folders(
    job_data_path, 
    subdirectories='latent_features',
    override=True)


# Training data - tensorflow data pipeline 
windows_train_path = os.path.join(job_data_path, 'windows/train')
train_files, num_train = dp.list_files_by_extension(windows_train_path, 'npy')
train_files = train_files[:train_data_limit]

train_dataset = m.create_tf_dataset_batched(
    train_files, 
    batch_size=cluster_batch_size, 
    buffer_size=cluster_buffer_size, 
    window_size=window_size,
    is_autoencoder=True, 
    shuffle=True)

# Load the trained autoencoder model
model_path = os.path.join(job_data_path,'model')
autoencoder_model = m.load_model(model_path, model_name=model_name)

# Path to save latent features to disk
latent_features_path = os.path.join(job_data_path, 'latent_features')

m.extract_latent_features_to_disk_from_prebatched_windows(
    autoencoder_model, 
    train_dataset, 
    latent_features_path, 
    features_name='latent_features_train',
    return_array=False,
    verbose=False)

# List and sort latent feature files
latent_features_path = os.path.join(job_data_path, 'latent_features')
latent_features_files, num_latent_files = dp.list_files_by_extension(latent_features_path, 'npy')

# Load the latent features from disk into a tensor dataset pipeline
latent_features_dataset = m.create_latent_features_tf_dataset(
    latent_features_files,
    batch_size=cluster_batch_size,
    shuffle=True, 
    shuffle_buffer_size=cluster_buffer_size)

cluster_model, convergence_history = m.train_kmeans(
    latent_features_dataset,                # tf.data.Dataset containing batches of latent feature vectors.
    batch_size=cluster_batch_size,          # Size of each batch for the KMeans model (controls memory usage and stability).
    num_clusters=num_clusters,              # Desired number of clusters (centroids) to form in the data.
    n_init=n_init,                          # Number of times the algorithm will run with different centroid seeds.
    max_iter=max_iter,                      # Maximum iterations allowed for each mini-batch to refine centroids.
    reassignment_ratio=reassignment_ratio   # Fraction of clusters reassigned per step; lower values stabilize updates.
)

# Save cluster model
cluster_model_path = os.path.join(job_data_path,'cluster_model')
m.save_cluster_model(cluster_model, cluster_model_path, model_name=cluster_model_name)

dp.delete_data_folders(
    job_data_path, 
    subdirectories='jpg',
    override=True)

# Prediction data
mtrx_predict_file_list, _ = dp.list_files_by_extension(mtrx_predict_path,'Z_mtrx',verbose=False)

data_scaling = 1.e9     # Scale the z-height of the data

flatten_method = 'row_and_poly_xy_deg_2'
flatten_method = 'row_mean'
flatten_method = 'iterate_mask'
flatten_method = 'poly_xy'

dp.process_mtrx_files(
    mtrx_predict_file_list,
    job_data_path, # save data path
    flatten_method = flatten_method, pixel_density = pixel_density, pixel_ratio = pixel_ratio,
    data_scaling = data_scaling, window_size = window_size, window_pitch = predict_window_pitch,
    save_windows = True,
    save_jpg = True,
    collate = False,
    verbose = False
    )

# Load the trained autoencoder
model_path = os.path.join(job_data_path,'model')
autoencoder_model = m.load_model(model_path, model_name=model_name)

# Load a previously saved cluster model from disk
cluster_model_path = os.path.join(job_data_path,'cluster_model')
cluster_model = m.load_cluster_model(cluster_model_path, model_name=cluster_model_name)

# Get list of folders containing predict window data
windows_predict_path = os.path.join(job_data_path, 'windows/predict')
predict_data_files_list = dp. find_bottom_layer_folders(windows_predict_path)
predict_data_files_list = predict_data_files_list[:predict_data_limit]

predictions_time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

for dir in predict_data_files_list:
    #Limit viewing to FU images
    if not dir.endswith("FU"):
        print('Not FU so skipping: {}'.format(dir))
        continue  

    # get list of the individual file names of the windows
    window_list, num_windows = dp.list_files_by_extension(dir, extension=".npy", verbose=False)
    # load the numpy windows
    image_windows = dp.load_n_numpy_files(window_list,num_windows)
    # load the coordinate files for the windows for window reconstruction
    image_windows_coordinates = dp.load_coordinates_file(dir)
    # Build the reconstruction of the original image
    reconstructed_img = dp.reconstruct_image(image_windows,image_windows_coordinates,window_size)

    # create a tf dataset from the numpy windows for making the latent features
    predict_dataset = m.create_tf_dataset(window_list, batch_size=num_windows, buffer_size=num_windows, shuffle=False, is_autoencoder=False)
    # make the latent features for each window using the autoencoder model 
    latent_predict_features, num_latent_predictions = m.extract_latent_features_to_disk_from_prebatched_windows(
        autoencoder_model, 
        predict_dataset, 
        latent_features_path, 
        features_name='latent_features_train',
        return_array=True,
        verbose=False)
    # make preductions 
    cluster_predictions = cluster_model.predict(latent_predict_features)
    # Build the reconstruction of the predicted cluster label data
    cluster_img = dp.reconstruct_cluster_image(image_windows_coordinates,window_size, cluster_predictions)
    image_name = os.path.basename(dir)
    # Path to save latent features to disk
    predictions_path = os.path.join(job_data_path, 'predictions')
    m.display_reconstructed_and_cluster_images(reconstructed_img,cluster_img,show_overlay=True,
                                                save_to_disk=True,
                                                predictions_time_stamp=predictions_time_stamp,
                                                output_path=job_data_path,
                                                image_name=image_name,
                                                dpi=150)

