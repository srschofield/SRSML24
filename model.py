#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML analysis of STM MTRX data

This file contains the model and associated functions
    
@author: Steven R. Schofield 

Created October 2024

"""

# ============================================================================
# Module dependencies
# ============================================================================

import sys
import platform
import os
import subprocess 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import io
from contextlib import redirect_stdout


import json
import pickle

import tensorflow as tf
from tensorflow.keras import layers

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib # for saving cluster model

from skimage.measure import label, regionprops

# Adam dependencies
import scipy 
import skimage as ski
from skimage import morphology, measure
from skimage import segmentation

import SRSML24.utils as ut
import SRSML24.data_prep as dp

# ============================================================================
# System information
# ============================================================================

def print_system_info():
    """
    Prints Python, TensorFlow, and system information, and checks if TensorFlow is using the GPU with CUDA.
    Also prints CPU and relevant GPU information.
    """
    # Print Python version
    print(f"\nPython version: {sys.version}")
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    # Check TensorFlow build information
    print(f"TensorFlow is built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"TensorFlow is built with ROCm: {tf.test.is_built_with_rocm()}")
    
    # Print system information
    print(f"\nSystem: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if TensorFlow is using GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nNumber of GPUs available to TensorFlow: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU Device: {gpu}")
    else:
        print("No GPU available, TensorFlow running on CPU")
    


    # Print summary of GPU availability
    if gpus:
        print("\n>>> Running with GPU available <<<  ({})\n".format(platform.platform()))
    else:
        print("\n@@@ NO GPU @@@  ({})\n".format(platform.platform()))



# ============================================================================
# Data IO
# ============================================================================

# Define a function to load and preprocess the numpy data
def load_numpy_file(file_path):
    """
    Load a numpy file and preprocess it by adding an additional channel.

    Args:
        file_path (tf.Tensor): Path to the .npy file.

    Returns:
        np.ndarray: Loaded and preprocessed numpy array.
    """
    try:
        file_path = file_path.numpy().decode('utf-8')  # Convert to string
        window = np.load(file_path)  # Load the numpy file
        window = np.expand_dims(window, axis=-1)  # Add channel dimension
        return window.astype(np.float32)  # Return as float32 for TensorFlow compatibility
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        #return np.zeros((32, 32, 1), dtype=np.float32)  # Return an empty array as a fallback


# Define a function to create a TensorFlow dataset from file paths
def create_tf_dataset(file_paths, batch_size=16, buffer_size=1000, is_autoencoder=True, shuffle=True):
    """
    Create a TensorFlow dataset from numpy file paths.

    Args:
        file_paths (list): List of paths to the numpy files.
        batch_size (int): The batch size for training/testing.
        buffer_size (int): The buffer size for shuffling.
        is_autoencoder (bool): Whether input data is also the target data (for autoencoders).

    Returns:
        tf.data.Dataset: The created TensorFlow dataset.
    """
    # Define a helper function to load numpy files
    def load_numpy_as_tensor(file_path):
        return tf.py_function(func=load_numpy_file, inp=[file_path], Tout=tf.float32)

    # Define a helper function for autoencoder mapping
    def autoencoder_map(x):
        return (x, x)
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_numpy_as_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # If autoencoder, input data is also the target data
    if is_autoencoder:
        dataset = dataset.map(autoencoder_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle, batch, and prefetch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    else:
        print('Not shuffling')

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Print details about the created pipeline
    print(f"Data pipeline created with {len(file_paths)} files, batch size: {batch_size}")
    for batch in dataset.take(1):  # Preview the first batch
        if is_autoencoder:
            print(f"Sample batch shape: {batch[0].shape}\n")
        else:
            print(f"Sample batch shape: {batch.shape}")

    return dataset



# Define a function to create a TensorFlow dataset from file paths (batched data)
def create_tf_dataset_batched(file_paths, batch_size=16, buffer_size=1000, is_autoencoder=True, shuffle=True, window_size=32):
    """
    Create a TensorFlow dataset from numpy file paths containing batches.

    Args:
        file_paths (list): List of paths to the numpy files containing batched data.
        batch_size (int): The batch size for training/testing.
        buffer_size (int): The buffer size for shuffling.
        is_autoencoder (bool): Whether input data is also the target data (for autoencoders).
        shuffle (bool): Whether to shuffle the data.
        window_size (int): The size of the square window.

    Returns:
        tf.data.Dataset: The created TensorFlow dataset.
    """

    # Define a function to load and preprocess batched numpy data
    def load_batched_numpy_file(file_path, window_size=32):
        """
        Load a batched numpy file, ensuring correct shape by adding a channel dimension if missing.

        Args:
            file_path (tf.Tensor): Path to the `.npy` file.
            window_size (int): The size of the square window.

        Returns:
            np.ndarray: Loaded and preprocessed batch of windows.
        """
        try:
            file_path = file_path.numpy().decode('utf-8')  # Convert to string
            batch = np.load(file_path)  # Load the numpy file

            # Check the shape and add the channel dimension if missing
            if batch.ndim == 3 and batch.shape[1:] == (window_size, window_size):  # Shape (N, window_size, window_size)
                batch = np.expand_dims(batch, axis=-1)  # Add channel dimension
            elif batch.ndim != 4 or batch.shape[1:] != (window_size, window_size, 1):
                raise ValueError(f"Invalid batch shape: {batch.shape}")

            return batch.astype(np.float32)  # Ensure float32 for TensorFlow
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            print('Please check that the data has window_size ({},{}). Possible mismatch with supplied data: {}'.format(window_size,window_size,batch.shape[1:]))
            return np.zeros((0, window_size, window_size, 1), dtype=np.float32)  # Empty batch

    # Define a helper function to load batched numpy files
    def load_numpy_as_tensor(file_path):
        return tf.py_function(func=lambda fp: load_batched_numpy_file(fp, window_size), inp=[file_path], Tout=tf.float32)

    # Define a helper function for autoencoder mapping
    def autoencoder_map(x):
        return (x, x)

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_numpy_as_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Flatten the dataset to treat each window individually
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    # If autoencoder, input data is also the target data
    if is_autoencoder:
        dataset = dataset.map(autoencoder_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle, batch, and prefetch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    else:
        print('Not shuffling')

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Print details about the created pipeline
    print(f"Data pipeline created with {len(file_paths)} files, batch size: {batch_size}, window size: {window_size}")
    for batch in dataset.take(1):  # Preview the first batch
        if is_autoencoder:
            print(f"Sample batch shape: {batch[0].shape}\n")
        else:
            print(f"Sample batch shape: {batch.shape}")

    return dataset



# ============================================================================
# Model IO
# ============================================================================


def save_model(model, model_path, model_name='autoencoder', model_train_time=''):
    """
    Saves the given model to the specified path, ensuring the directory exists.

    Parameters:
    - model: The trained model you want to save.
    - model_path: The directory where you want to save the model.
    - model_name: The name of the model file (without extension). Default is 'autoencoder'.
    """
    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Save the model
    model_name_output = f"{model_name}.keras"
    model_file_path = os.path.join(model_path, model_name_output)
    model.save(model_file_path)
    print(f"Model saved at: {model_file_path}")

    # Add date/time to the model name
    if model_train_time != '':
        model_name_output = f"{model_name}_{model_train_time}.keras"
        model_file_path = os.path.join(model_path, model_name_output)
        model.save(model_file_path)
        print(f"Backup of the model with time stamp saved at model saved at: {model_file_path}")

    



def load_model(model_path, model_name='autoencoder', compile_model=False):
    """
    Loads the specified model from the given path without compiling it.

    Parameters:
    - model_path: The directory where the model is saved.
    - model_name: The name of the model file (without extension). Default is 'autoencoder'.
    - compile_model: If False, loads the model without compiling it (default behavior).
    
    Returns:
    - The loaded model.
    """
    # Full model file path with .keras extension
    model_file_path = os.path.join(model_path, model_name + '.keras')
    
    # Load the model without compiling it
    loaded_model = tf.keras.models.load_model(model_file_path, compile=compile_model)
    print(f"Model loaded from: {model_file_path}")
    
    return loaded_model


# def save_history(history, model_path, model_name='autoencoder', 
#                  loss_name='loss', val_loss_name='val_loss', 
#                  metrics=None, val_metrics=None,
#                  model_name='', 
#                  model_train_time=''):
#     """
#     Saves the training history to a text file in the same directory as the model.
    
#     Parameters:
#     - history: The history object returned by the `model.fit()` method.
#     - model_path: The directory where the model and history will be saved.
#     - model_name: The name of the history file (without extension). Default is 'autoencoder'.
#     - loss_name: The name of the training loss metric in the history. Default is 'loss'.
#     - val_loss_name: The name of the validation loss metric in the history. Default is 'val_loss'.
#     - metrics: A list of names of training metrics to be included in the history. Default is None.
#     - val_metrics: A list of names of validation metrics to be included in the history. Default is None.
#     """
#     # Ensure the directory exists, create it if it doesn't
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)

#     # Add date/time to the history file name
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     history_path = os.path.join(model_path, f"{model_name}_history_{timestamp}.txt")

#     # Extract history data
#     epochs = range(1, len(history.history[loss_name]) + 1)
#     train_loss = history.history[loss_name]
#     val_loss = history.history.get(val_loss_name, [])

#     # Open the file for writing
#     with open(history_path, 'w') as f:
#         # Write headers (loss and optional metrics)
#         headers = f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}"
        
#         # If metrics are provided, add them to the headers
#         if metrics:
#             for metric in metrics:
#                 headers += f"{'Train ' + metric:<15}"
#         if val_metrics:
#             for val_metric in val_metrics:
#                 headers += f"{'Val ' + val_metric:<15}"
#         f.write(headers + "\n")

#         # Write values for each epoch
#         for epoch in epochs:
#             # Basic loss values
#             line = f"{epoch:<10}{train_loss[epoch-1]:<15.5f}{val_loss[epoch-1] if val_loss else '':<15.5f}"
            
#             # Add train and validation metrics, if provided
#             if metrics:
#                 for metric in metrics:
#                     line += f"{history.history[metric][epoch-1]:<15.5f}"
#             if val_metrics:
#                 for val_metric in val_metrics:
#                     line += f"{history.history[val_metric][epoch-1]:<15.5f}"
            
#             f.write(line + "\n")

#     print(f"History saved to {history_path}")



# ============================================================================
# Model - UNET
# ============================================================================


def build_autoencoder(window_size, model_name='autoencoder'):
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Define the input shape
    inputs = tf.keras.Input(shape=(window_size, window_size, 1), name='input')
    
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='conv1')(inputs)
    d1 = layers.Dropout(0.1, name='drop1')(c1)
    p1 = layers.MaxPooling2D(pool_size=2, name='pool1')(d1)
    
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='conv2')(p1)
    d2 = layers.Dropout(0.1, name='drop2')(c2)
    p2 = layers.MaxPooling2D(pool_size=2, name='pool2')(d2)
    
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='conv3')(p2)
    d3 = layers.Dropout(0.1, name='drop3')(c3)
    p3 = layers.MaxPooling2D(pool_size=2, name='pool3')(d3)
    
    # Bottleneck
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='bottleneck')(p3)
    
    # Decoder
    u1 = layers.UpSampling2D(size=2, name='up1')(c4)
    u1 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='upconv1')(u1)
    u1 = layers.concatenate([u1, c3], axis=-1, name='skip1')
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='conv4')(u1)
    
    u2 = layers.UpSampling2D(size=2, name='up2')(c5)
    u2 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='upconv2')(u2)
    u2 = layers.concatenate([u2, c2], axis=-1, name='skip2')
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='conv5')(u2)
    
    u3 = layers.UpSampling2D(size=2, name='up3')(c6)
    u3 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='upconv3')(u3)
    u3 = layers.concatenate([u3, c1], axis=-1, name='skip3')
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name='conv6')(u3)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same', name='output')(c7)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def save_model_summary(model, model_path, model_name):
    os.makedirs(model_path, exist_ok=True)
    parts = [model_name, 'model_summary']
    file_base = '_'.join([p for p in parts if p])
    file_name = f"{file_base}.txt"
    file_path = os.path.join(model_path, file_name)
    
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def save_model_diagram(model, model_path, model_name='model',
                       show_shapes=False, show_layer_names=False, expand_nested=False):
    """
    Saves a PNG diagram of the Keras model architecture using plot_model.
    Falls back gracefully if pydot or Graphviz are not installed.
    
    Parameters:
        model (tf.keras.Model): The model to visualize.
        model_path (str): Directory to save the image.
        model_name (str): Base name to use in the output filename.
        show_shapes (bool): Whether to show output shapes on the diagram.
        show_layer_names (bool): Whether to show layer names on the diagram.
        expand_nested (bool): Whether to expand nested models (e.g. submodules).
    """
    try:
        from tensorflow.keras.utils import plot_model
        os.makedirs(model_path, exist_ok=True)
        file_path = os.path.join(model_path, f"{model_name}_model_summary.png")
        
        plot_model(model,
                   to_file=file_path,
                   show_shapes=show_shapes,
                   show_layer_names=show_layer_names,
                   expand_nested=expand_nested)
        
        print(f"[Info] Model diagram saved to {file_path}")
        
    except (ImportError, OSError) as e:
        print(f"[Warning] Could not generate model diagram. Reason: {e}")
        print("To enable this feature, install:\n"
              "  pip install pydot graphviz\n"
              "  brew install graphviz  # or apt install graphviz")



# ============================================================================
# Model training information
# ============================================================================

def save_history(history, model_name, model_path, model_train_time=''):
    """
    Saves the Keras History or a history-like dict to disk as a .dat (pickle) file.
    Filename: '{model_name}_{model_train_time}_history_data.dat', omitting empty parts.

    Parameters:
    -----------
    history: History object or dict
        If a Keras History, its `.history` attribute will be used. If it's already a dict,
        it will be saved directly.
    model_name: str
        Identifier for the model, used in the filename.
    model_path: str
        Directory where the file will be saved.
    model_train_time: str, optional
        Timestamp or descriptor to include in the filename.

    Returns:
    --------
    str: The full path to the saved .dat file.
    """
    # Extract raw data dict
    data = history.history if hasattr(history, 'history') else history
    # Ensure directory exists
    os.makedirs(model_path, exist_ok=True)
    # Build filename parts, skipping empty
    parts = [model_name, model_train_time, 'history_data']
    file_base = '_'.join([p for p in parts if p])
    file_name = f"{file_base}.dat"
    file_path = os.path.join(model_path, file_name)
    # Save via pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    # Notify user
    print(f"The training history data has been saved to disk as a binary pickle file in: {file_path}")
    return file_path


def plot_history_from_file(file_path,
                           loss_name='loss',
                           val_loss_name='val_loss',
                           metric_names=None,
                           val_metric_names=None,
                           dpi=150,
                           show_plot=True):
    """
    Loads training history from a .dat (pickle) or .json file and plots loss and metrics,
    optionally showing and saving the figure as a JPG.

    Parameters:
    -----------
    file_path: str
        Path to the history file (.dat or .json).
    loss_name: str
        Key for training loss. Default 'loss'.
    val_loss_name: str
        Key for validation loss. Default 'val_loss'.
    metric_names: list of str
        Training metric keys. Defaults to ['mse', 'mae'] if None.
    val_metric_names: list of str
        Validation metric keys. Defaults to ['val_mse', 'val_mae'] if None.
    model_name: str
        Optional identifier for output filename.
    model_train_time: str
        Optional timestamp for output filename.
    output_dir: str
        Directory to save the plot JPG. If None, saves next to history file.
    dpi: int
        Resolution for saved figure.
    show_plot: bool
        If True, calls plt.show() to display the figure.

    Returns:
    --------
    str: Path where the plot was saved.
    """
    # Load history
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.dat':
        with open(file_path, 'rb') as f:
            history = pickle.load(f)
    elif ext.lower() == '.json':
        with open(file_path, 'r') as f:
            history = json.load(f)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # Defaults
    if metric_names is None:
        metric_names = ['mse', 'mae']
    if val_metric_names is None:
        val_metric_names = ['val_mse', 'val_mae']

    # Extract data series
    loss = history.get(loss_name)
    val_loss = history.get(val_loss_name)
    metrics = {m: history.get(m) for m in metric_names}
    val_metrics = {vm: history.get(vm) for vm in val_metric_names}
    epochs = range(1, len(loss) + 1) if loss else []

    # Plot
    plt.figure(figsize=(12, 5))
    # Loss subplot
    if loss and val_loss:
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    else:
        print("Warning: Missing loss or val_loss in history.")
    # Metrics subplot
    plt.subplot(1, 2, 2)
    for m, vm in zip(metric_names, val_metric_names):
        tr = metrics.get(m)
        vl = val_metrics.get(vm)
        if tr and vl:
            plt.plot(epochs, tr, label=f'Train {m}')
            plt.plot(epochs, vl, linestyle='--', label=f'Val {m}')
    plt.title('Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    # Save before displaying or closing
    out_path = os.path.splitext(file_path)[0] + '.jpg'
    plt.savefig(out_path, format='jpg', dpi=dpi)

    if show_plot:
        plt.show()

    plt.close()
    return out_path

# ============================================================================
# Model testing
# ============================================================================

# Function to predict and visualize the original and reconstructed image
def visualize_reconstruction(model, test_image):
    # Ensure the test image is in the correct shape: (window_size, window_size)
    # Autoencoder expects a batch, so we need to add a dimension to the image: (1, 32, 32, 1)
    test_image = np.expand_dims(test_image, axis=0)

    # Predict using the autoencoder
    reconstructed_image = model.predict(test_image)

    # Remove the batch dimension for visualization
    original = np.squeeze(test_image)
    reconstructed = np.squeeze(reconstructed_image)

    # Plot the original and the reconstructed images side by side
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_reconstructed_and_cluster_images(reconstructed_img, cluster_img, show_overlay=True, 
                                             save_to_disk=False, output_path=None, image_name='img', dpi=150):
    """
    Display side-by-side images: the reconstructed input image, the cluster labels,
    and optionally an overlay of the reconstructed image with cluster labels on top.

    Parameters:
    - reconstructed_img (ndarray): The reconstructed image, typically the output
      from an autoencoder or similar model.
    - cluster_img (ndarray): The image showing cluster labels, usually resulting
      from clustering analysis on the latent space.
    - show_overlay (bool): Optional; if True, displays a third panel with an overlay
      of the reconstructed image and cluster labels. Default is True.
    - save_to_disk (bool): Optional; if True, saves the image to disk instead of displaying it. Default is False.
    - output_path (str): The path to save the image if save_to_disk is True. Required if save_to_disk is True.
    - dpi (int): Optional; the resolution in dots per inch for saving the image. Default is 300.

    Returns:
    - None: This function either displays the plot or saves it to disk based on save_to_disk.
    """

    # Output directory with timestamp
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory created: {output_path}")

    # Determine the number of subplots based on show_overlay
    n_plots = 3 if show_overlay else 2
    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

    # Plot the first image (Reconstructed Image)
    ax[0].imshow(reconstructed_img, cmap='viridis')
    ax[0].set_title('Input Image')
    ax[0].axis('off')  # Remove axis labels

    # Plot the second image (Cluster Labels)
    ax[1].imshow(cluster_img, cmap='turbo')
    ax[1].set_title('Cluster Labels')
    ax[1].axis('off')  # Remove axis labels

    # Optionally plot the third image (Overlay of Reconstructed Image with Cluster Labels)

    if show_overlay:
        overlay_cluster_with_alpha(ax[2], reconstructed_img, cluster_img)

    plt.tight_layout()
    
    # Save to disk or display
    if save_to_disk:
        if output_path:
            output_path = os.path.join(output_path, f"{image_name}.jpg")
            plt.savefig(output_path, format='jpg', dpi=dpi)
            print(f"Image saved to {output_path} with dpi={dpi}")
        else:
            print("Error: output_path must be specified when save_to_disk is True.")
    else:
        plt.show()

    # Close the figure to release memory
    plt.close(fig)


def overlay_cluster_with_alpha(ax, reconstructed_img, cluster_img, overlay_cmap='turbo', alpha=0.6):
    ax.imshow(reconstructed_img, cmap='gray')  # or 'viridis'
    ax.imshow(cluster_img, cmap=overlay_cmap, alpha=alpha)  # no transparency logic
    ax.set_title("Overlayed Image with Cluster Labels")
    ax.axis('off')

def analyse_cluster_labels(cluster_image, large_region_thresh=0.1):
    label_stats = {}
    background_labels = []

    image_area = cluster_image.shape[0] * cluster_image.shape[1]

    for cluster_id in np.unique(cluster_image):
        mask = (cluster_image == cluster_id)
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        region_sizes = [r.area for r in regions]
        total_area = np.sum(mask)
        max_region_size = max(region_sizes) if region_sizes else 0
        is_background = max_region_size > large_region_thresh * image_area

        label_stats[cluster_id] = {
            'total_area': total_area,
            'percent_area': 100 * total_area / image_area,
            'num_regions': len(region_sizes),
            'mean_region_size': np.mean(region_sizes) if region_sizes else 0,
            'max_region_size': max_region_size,
            'max_region_percent': 100 * max_region_size / image_area,
            'is_background': is_background,
        }

        if is_background:
            background_labels.append(cluster_id)

    return label_stats, background_labels


def relabel_background(cluster_img, background_labels):
    # Create a copy to avoid modifying the original
    new_img = np.copy(cluster_img)

    # Set all background label pixels to -1
    for bg_label in background_labels:
        new_img[cluster_img == bg_label] = -1

    # Shift non-background labels by +1
    mask_non_background = new_img >= 0
    new_img[mask_non_background] += 1

    return new_img

# ============================================================================
# Clustering
# ============================================================================

# def extract_latent_features_to_disk(autoencoder_model, dataset, features_path, features_name='latent_features', batches_per_file=500):
#     """
#     Extracts and saves latent features from a TensorFlow dataset to disk in groups of batches, minimizing memory usage.

#     Parameters:
#     - autoencoder_model: The trained autoencoder model to extract latent features.
#     - dataset: The TensorFlow dataset from which to extract latent features.
#     - features_path: The directory where latent features should be saved.
#     - features_name: The base name for the saved feature files.
#     - batches_per_file: Number of batches to accumulate in memory before saving to a single file.

#      Structure of Saved Data:
#     - Each saved file is a .npy file containing a 2D NumPy array.
#     - Shape: (batches_per_file * batch_size, latent_dim), where:
#         - batch_size is the number of samples per batch in the dataset.
#         - latent_dim is the dimensionality of the bottleneck layer in the autoencoder.
#     - Each row represents the latent feature vector for a single sample.
#     """
#     # Create encoder model to extract latent space from 'Bottleneck' layer
#     encoder_model = tf.keras.Model(
#         inputs=autoencoder_model.input, 
#         outputs=autoencoder_model.get_layer('Bottleneck').output
#     )
    
#     if not os.path.exists(features_path):
#         os.makedirs(features_path)

#     # Get the total number of batches in the dataset
#     total_batches = dataset.cardinality().numpy()

#     # Check if batches_per_file is less than the total number of batches
#     if batches_per_file > total_batches:
#         print(f"Error: 'batches_per_file' ({batches_per_file}) must be less than the total number of batches in the dataset ({total_batches}).\n Setting batches_per_file=total_batches")
#         batches_per_file = total_batches

#     # Print initial statement about batch processing
#     print(f"Saving latent features in groups of {batches_per_file} batches per file.")
#     print(f"There are {total_batches} total batches in the dataset.")
    
#     # Initialize a list to hold accumulated batches
#     accumulated_batches = []
#     file_index = 0
    
#     # Process each batch and accumulate them
#     for i, batch in enumerate(dataset):
#         inputs = batch[0]  # Extract inputs, ignoring labels if present
#         latent_features_batch = encoder_model.predict(inputs, verbose=0)
        
#         # Flatten latent features to 2D (if necessary) and add to accumulated batches
#         latent_features_batch_flat = latent_features_batch.reshape(latent_features_batch.shape[0], -1)
#         accumulated_batches.append(latent_features_batch_flat)

#         if batches_per_file >= 100:
#             if (i+1) % 100 == 0:
#                 print(i+1)
#             else:
#                 print(".",end="")
#         else:
#             if (i+1) % batches_per_file == 0:
#                 print(i+1)
#             else:
#                 print(".",end="")

#         # When enough batches have been accumulated, save to a single file
#         if (i + 1) % batches_per_file == 0:
#             # Concatenate accumulated batches and save to disk
#             file_path = os.path.join(features_path, f"{features_name}_{file_index}.npy")
#             np.save(file_path, np.concatenate(accumulated_batches, axis=0))
#             print(f"\nSaved {batches_per_file} batches to {file_path}")

#             # Clear accumulated batches and increment file index
#             accumulated_batches = []
#             file_index += 1
    
#     # Save any remaining batches that didn't reach the batch limit
#     if accumulated_batches:
#         file_path = os.path.join(features_path, f"{features_name}_{file_index}.npy")
#         np.save(file_path, np.concatenate(accumulated_batches, axis=0))
#         print(f"Saved remaining batches to {file_path}")
    
#     print(f"\nLatent features have been saved to {features_path} in grouped batch files.")

def extract_latent_features_to_disk_from_prebatched_windows(
    autoencoder_model,
    dataset,
    features_path,
    bottleneck_layer_name='bottleneck',
    features_name='latent_features',
    return_array=False,
    verbose=False):
    """
    Extracts latent features from a pre-batched TensorFlow dataset, where each batch is processed
    and saved directly as a corresponding batch of latent features.

    Parameters:
    - autoencoder_model: The trained autoencoder model to extract latent features.
    - dataset: The TensorFlow dataset containing pre-batched input windows.
    - features_path: The directory where latent features should be saved.
    - bottleneck_layer_name: The name of the bottleneck layer in the autoencoder model.
    - features_name: The base name for the saved feature files.
    - verbose: Whether to print progress information.

    Structure of Saved Data:
    - Each batch is saved as a .npy file containing a 2D NumPy array.
    - Shape: (batch_size, latent_dim), where:
        - batch_size is the number of samples in the batch.
        - latent_dim is the flattened size of the bottleneck layer.
    - Each row represents the latent feature vector for a single input window.
    """
    # Create encoder model to extract latent space from the specified bottleneck layer
    encoder_model = tf.keras.Model(
        inputs=autoencoder_model.input,
        outputs=autoencoder_model.get_layer(bottleneck_layer_name).output
    )

    if return_array:
        latent_features_all = []
    else:
        # Ensure the output directory exists
        if not os.path.exists(features_path):
            os.makedirs(features_path)

    # Process each batch in the dataset
    for i, batch in enumerate(dataset):
        inputs = batch if isinstance(batch, tf.Tensor) else batch[0]  # Handle datasets with or without labels
        batch_shape = tf.shape(inputs).numpy()
        if verbose: 
            print(f"Processing batch {i + 1}, input shape: {batch_shape}")
        
        # Compute latent features for the current batch
        latent_features_batch = encoder_model.predict(inputs, verbose=0)
        
        # Flatten latent features if necessary
        if len(latent_features_batch.shape) > 2:
            flattened_latent_features = tf.reshape(
                latent_features_batch, 
                [latent_features_batch.shape[0], -1]
            ).numpy()
        else:
            flattened_latent_features = latent_features_batch
        
        # Validate output shape (N, M)
        latent_shape = flattened_latent_features.shape

        if i==0: 
            sample_batch_shape = flattened_latent_features.shape

        if verbose:
            print(f"Latent features shape for batch {i + 1}: {latent_shape}")
        else:
            if i>0 and i%100==0:
                print(i,)
            else:
                print('.',end='')
        
        if return_array:
            latent_features_all.append(flattened_latent_features)
        else:
            # Save the latent features batch to disk
            file_path = os.path.join(features_path, f"{features_name}_batch_{i}.npy")
            np.save(file_path, flattened_latent_features)
            if verbose:
              print(f"Saved latent features for batch {i + 1} to {file_path}")
    
    if return_array:
        # Combine all latent features into a single array
        latent_features_all = np.concatenate(latent_features_all, axis=0)
        print(f"Combined latent features shape: {latent_features_all.shape}")
        return latent_features_all, latent_features_all.shape[0]
    else:
        print(f"\nAll latent features have been saved to {features_path}.")
        print(f"Sample batch shape: {sample_batch_shape}")


# def extract_latent_features_to_array(autoencoder_model, input_data):
#     """
#     Extracts latent features from an individual input (or batch of inputs) and returns them as a NumPy array.

#     Parameters:
#     - autoencoder_model: The trained autoencoder model to extract latent features.
#     - input_data: A single input or a batch of inputs for which to extract latent features.

#     Returns:
#     - np.ndarray: A 2D NumPy array containing the latent feature vectors.
#                   Shape: (num_samples, latent_dim), where num_samples is the number of samples in input_data.
#     """
#     # Create encoder model to extract latent space from 'Bottleneck' layer
#     encoder_model = tf.keras.Model(
#         inputs=autoencoder_model.input,
#         outputs=autoencoder_model.get_layer('Bottleneck').output
#     )
    
#     # Predict latent features for the input data
#     latent_features = encoder_model.predict(input_data, verbose=0)
    
#     # Flatten latent features to 2D if necessary
#     latent_features_flat = latent_features.reshape(latent_features.shape[0], -1)
    
#     num_features = len(latent_features_flat)
#     return latent_features_flat, num_features


def create_latent_features_tf_dataset(latent_feature_files, batch_size=32, shuffle=False, shuffle_buffer_size=1000):
    """
    Creates a TensorFlow data pipeline that reads latent feature files in batches and prepares them
    for clustering or further analysis. The pipeline reads each `.npy` file containing latent features,
    loads the data as a NumPy array, and processes it in batches.

    Parameters:
    -----------
    latent_feature_files : list of str
        A list of file paths to `.npy` files containing latent feature arrays. Each file is expected to contain
        a NumPy array with shape (N, M), where N is the number of latent features, and M is the feature length.
    batch_size : int, optional
        The number of latent feature vectors to include in each batch. Default is 32.
    shuffle : bool, optional
        If True, shuffles the dataset using a specified buffer size. Default is False.
    shuffle_buffer_size : int, optional
        The buffer size for shuffling. Default is 1000.

    Returns:
    --------
    tf.data.Dataset
        A TensorFlow Dataset object that yields batches of latent features, each with shape
        (batch_size, M) where M is the feature length. The dataset is configured to prefetch batches
        for optimized loading performance.
    """
    def load_numpy_file(file_path):
        data = np.load(file_path.numpy().decode('utf-8'))
        return data

    def tf_load_numpy_file(file_path):
        data = tf.py_function(load_numpy_file, [file_path], tf.float32)
        return tf.data.Dataset.from_tensor_slices(data)

    # Create a dataset from file paths
    file_dataset = tf.data.Dataset.from_tensor_slices(latent_feature_files)
    
    # Shuffle files if specified
    if shuffle:
        file_dataset = file_dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Map the function to load numpy data using `interleave`
    data_pipeline = file_dataset.interleave(
        tf_load_numpy_file,
        cycle_length=len(latent_feature_files),
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch the data
    data_pipeline = data_pipeline.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Print details about the created pipeline
    print(f"Data pipeline created with {len(latent_feature_files)} files, batch size: {batch_size}")
    if shuffle:
        print(f"Shuffling enabled with buffer size: {shuffle_buffer_size}")
    for batch in data_pipeline.take(1):  # Preview the first batch
        print(f"Batch shape: {batch.shape}")
    
    return data_pipeline


# def create_latent_features_tf_dataset(latent_feature_files, batch_size=32):
#     """
#     Creates a TensorFlow data pipeline that reads latent feature files in batches and prepares them
#     for clustering or further analysis. The pipeline reads each `.npy` file containing latent features,
#     loads the data as a NumPy array, and processes it in batches.

#     Parameters:
#     -----------
#     latent_feature_files : list of str
#         A list of file paths to `.npy` files containing latent feature arrays. Each file is expected to contain
#         a NumPy array with shape (N, M), where N is the number of latent features, and M is the feature length.
#     batch_size : int, optional
#         The number of latent feature vectors to include in each batch. Default is 32.

#     Returns:
#     --------
#     tf.data.Dataset
#         A TensorFlow Dataset object that yields batches of latent features, each with shape
#         (batch_size, M) where M is the feature length. The dataset is configured to prefetch batches
#         for optimized loading performance.
#     """
#     def load_numpy_file(file_path):
#         data = np.load(file_path.numpy().decode('utf-8'))
#         return data

#     def tf_load_numpy_file(file_path):
#         data = tf.py_function(load_numpy_file, [file_path], tf.float32)
#         return tf.data.Dataset.from_tensor_slices(data)

#     # Create a dataset from file paths
#     file_dataset = tf.data.Dataset.from_tensor_slices(latent_feature_files)
    
#     # Map the function to load numpy data using `interleave`
#     data_pipeline = file_dataset.interleave(
#         tf_load_numpy_file,
#         cycle_length=len(latent_feature_files),
#         block_length=1,
#         num_parallel_calls=tf.data.AUTOTUNE
#     )

#     # Batch and prefetch the data
#     data_pipeline = data_pipeline.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     # Print details about the created pipeline
#     print(f"Data pipeline created with {len(latent_feature_files)} files, batch size: {batch_size}")
#     for batch in data_pipeline.take(1):  # Preview the first batch
#         print(f"Batch shape: {batch.shape}")
    
#     return data_pipeline


# def create_latent_features_tf_dataset_from_array(latent_features_array, batch_size=32):
#     """
#     Creates a TensorFlow data pipeline from a NumPy array of latent features, preparing them
#     for clustering or further analysis in batches.

#     Parameters:
#     -----------
#     latent_features_array : np.ndarray
#         A NumPy array containing latent features, with shape (N, M), where N is the number of samples,
#         and M is the dimensionality of the latent features.
#     batch_size : int, optional
#         The number of latent feature vectors to include in each batch. Default is 32.

#     Returns:
#     --------
#     tf.data.Dataset
#         A TensorFlow Dataset object that yields batches of latent features, each with shape
#         (batch_size, M), where M is the dimensionality of the features. The dataset is configured to
#         prefetch batches for optimized loading performance.
#     """
#     # Ensure the input is a NumPy array
#     if not isinstance(latent_features_array, np.ndarray):
#         raise ValueError("latent_features_array must be a NumPy array")

#     # Convert the NumPy array to a TensorFlow dataset
#     data_pipeline = tf.data.Dataset.from_tensor_slices(latent_features_array)
    
#     # Batch and prefetch the data
#     data_pipeline = data_pipeline.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     # Print details about the created pipeline
#     print(f"Data pipeline created from NumPy array with shape {latent_features_array.shape}, batch size: {batch_size}")
   
#     for batch in data_pipeline.take(1):  # Preview the first batch
#         print(f"Batch shape: {batch.shape}")
    
#     return data_pipeline


def train_kmeans(data_pipeline, batch_size=2048, num_clusters=10, n_init=1, max_iter=300, reassignment_ratio=0.01):
    """
    Trains a MiniBatchKMeans clustering model using latent feature data from a TensorFlow data pipeline.
    Adds optimizations for improved fitting in batch-based training, including a warm start with KMeans.

    Parameters:
    -----------
    data_pipeline : tf.data.Dataset
        A TensorFlow Dataset object yielding batches of latent feature vectors, typically created using
        `create_latent_features_tf_dataset`.
    batch_size : int, optional
        The number of samples per batch to be used for each partial fit in MiniBatchKMeans.
        Larger batch sizes can stabilize updates but may require more memory. Default is 2048.
    num_clusters : int, optional
        The number of clusters (centroids) to form. Default is 10.
    n_init : int, optional
        Number of times the KMeans algorithm will run with different centroid seeds. The final results
        will be the best output among these initializations, selected based on the inertia.
        Higher values improve clustering robustness but increase computation time. Default is 10.
    max_iter : int, optional
        Maximum number of iterations over each mini-batch to perform in the KMeans optimization loop.
        Higher values allow more refinement per batch but increase runtime. Default is 300.
    reassignment_ratio : float, optional
        The fraction of clusters that are reassigned to new values based on incoming data.
        Lower values lead to smoother centroid updates, which can improve convergence stability.
        Range: 0 < reassignment_ratio < 1. Default is 0.01.

    Returns:
    --------
    MiniBatchKMeans
        A trained MiniBatchKMeans model fitted to the latent features from the dataset.
    list of float
        A list of inertia values for each batch, tracking convergence across iterations.
    """
    # Warm start by fitting KMeans on a small subset to initialize centroids
    subset = np.concatenate([batch.numpy() for batch in data_pipeline.take(10)])  # Example subset of 10 batches
    initial_kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, init='k-means++').fit(subset)
    
    # Initialize MiniBatchKMeans with precomputed centroids and specified n_init
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        n_init=1,
        max_iter=max_iter,
        reassignment_ratio=reassignment_ratio,
        init=initial_kmeans.cluster_centers_
    )
    
    convergence_history = []

    for i, batch in enumerate(data_pipeline):
        # Convert batch to NumPy array if needed
        latent_features_batch = np.array(batch) if isinstance(batch, tf.Tensor) else batch
        latent_features_batch = latent_features_batch.reshape(-1, latent_features_batch.shape[-1])

        # Perform partial fitting with the current batch
        kmeans.partial_fit(latent_features_batch)

        # Track inertia for convergence monitoring
        inertia = kmeans.inertia_
        convergence_history.append(inertia)
        print(f"Batch {i+1} processed. Inertia: {inertia}")
        sys.stdout.flush()

    return kmeans, convergence_history


def plot_kmeans_convergence(convergence_history, cluster_model_path, model_name, dpi=150):
    """
    Plots MiniBatchKMeans convergence over batches, excluding the final point
    which may correspond to a smaller, less representative batch. Saves both
    the plot and the raw inertia values (as plain text).

    Parameters:
        convergence_history (list): Inertia values recorded per batch.
        cluster_model_path (str): Directory to save output files.
        model_name (str): Base name for saved files.
        dpi (int): Dots per inch for saved figure.
    """
    if len(convergence_history) <= 1:
        print("[Warning] Not enough data to plot convergence.")
        return

    # Exclude final point
    history_to_plot = convergence_history[:-1]

    # Ensure output directory exists
    os.makedirs(cluster_model_path, exist_ok=True)

    # Define save paths
    image_path = os.path.join(cluster_model_path, f"{model_name}_convergence.jpg")
    txt_path = os.path.join(cluster_model_path, f"{model_name}_convergence.txt")

    # Save raw inertia values as plain text (one per line)
    with open(txt_path, 'w') as f:
        for val in history_to_plot:
            f.write(f"{val}\n")
    print(f"[Info] Raw convergence data saved to {txt_path}")

    # Plot
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(history_to_plot) + 1))
    plt.plot(epochs, history_to_plot, marker='o', linewidth=1.5)
    plt.title("MiniBatchKMeans Convergence")
    plt.xlabel("Batch Index")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(image_path, dpi=dpi)
    print(f"[Info] Convergence plot saved to {image_path}")
    plt.show()


def evaluate_and_visualize_clustering(latent_features, cluster_labels):
    """
    Evaluate clustering with silhouette score and visualize the clusters using PCA.

    Parameters:
    - latent_features: numpy array, the features that were clustered.
    - cluster_labels: numpy array, the labels assigned to each data point by the clustering algorithm.
    - title: str, the title for the PCA visualization plot.
    """
    # Calculate and print silhouette score
    print('Calculating silhouette score: ', end="")
    silhouette_avg = silhouette_score(latent_features, cluster_labels)
    print(silhouette_avg)

    # Visualize the clusters using PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_features)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.title("Cluster Visualization (PCA)")
    plt.show()


def gather_latent_features(data_pipeline):
    """
    Collects and concatenates latent features from a TensorFlow data pipeline into a single NumPy array.

    This function iterates over a TensorFlow `data_pipeline` (typically generated from saved .npy files of
    latent features), converts each batch to a NumPy array, flattens it to ensure a consistent two-dimensional
    shape, and then appends it to a list. Finally, it concatenates all batches into one array along the first axis.

    Parameters:
    -----------
    data_pipeline : tf.data.Dataset
        A TensorFlow Dataset object containing batches of latent feature vectors, each batch as a NumPy array.

    Returns:
    --------
    np.ndarray
        A single concatenated NumPy array of all latent features with shape (total_samples, feature_length),
        where `total_samples` is the total number of latent feature vectors across all batches, and
        `feature_length` is the dimensionality of each latent feature vector.
    """
    all_latent_features = []
    for batch in data_pipeline:
        latent_features_batch = batch.numpy()  # Convert to numpy array
        latent_features_batch = latent_features_batch.reshape(-1, latent_features_batch.shape[-1])  # Flatten batch
        all_latent_features.append(latent_features_batch)
    return np.concatenate(all_latent_features, axis=0)


# ============================================================================
# Cluster IO
# ============================================================================

def save_cluster_model(cluster_model, model_path, model_name='cluster_model'):
    """
    Saves the given clustering model to the specified path, ensuring the directory exists.

    Parameters:
    - cluster_model: The trained clustering model you want to save.
    - model_path: The directory where you want to save the model.
    - model_name: The name of the model file (without extension). Default is 'cluster_model'.
    """
    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Full model file path with .pkl extension
    model_file_path = os.path.join(model_path, model_name + '.pkl')

    # Save the clustering model using joblib
    joblib.dump(cluster_model, model_file_path)
    print(f"Cluster model saved at: {model_file_path}")


def load_cluster_model(model_path, model_name='cluster_model'):
    """
    Loads the specified clustering model from the given path.

    Parameters:
    - model_path: The directory where the model is saved.
    - model_name: The name of the model file (without extension). Default is 'cluster_model'.
    
    Returns:
    - The loaded clustering model.
    """
    # Full model file path with .pkl extension
    model_file_path = os.path.join(model_path, model_name + '.pkl')
    
    # Load the clustering model using joblib
    loaded_model = joblib.load(model_file_path)
    print(f"Cluster model loaded from: {model_file_path}")
    
    return loaded_model


def save_latent_features(latent_features, features_path, features_name='latent_features'):
    """
    Saves the given latent features to the specified path in smaller chunks, ensuring the directory exists.

    Parameters:
    - latent_features: The array of latent features to save.
    - features_path: The directory where you want to save the latent features.
    - features_name: The name of the file to save the features in (without extension). Default is 'latent_features'.
    """
    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    # Full latent features file path with .npy extension
    features_file_path = os.path.join(features_path, features_name + '.npy')

    # Save latent features in chunks
    chunk_size = 1000  # Adjust as needed
    with open(features_file_path, 'wb') as f:
        for i in range(0, latent_features.shape[0], chunk_size):
            np.save(f, latent_features[i:i + chunk_size])
    print(f"Latent features saved at: {features_file_path}")

def load_latent_features(features_path, features_name='latent_features'):
    """
    Loads the latent features from the specified path.

    Parameters:
    - features_path: The directory where the latent features are saved.
    - features_name: The name of the file to load the features from (without extension). Default is 'latent_features'.

    Returns:
    - The loaded latent features as a NumPy array.
    """
    # Full latent features file path with .npy extension
    features_file_path = os.path.join(features_path, features_name + '.npy')

    # Load latent features in chunks
    latent_features_list = []
    with open(features_file_path, 'rb') as f:
        while True:
            try:
                latent_features_list.append(np.load(f))
            except ValueError:
                break
    latent_features = np.concatenate(latent_features_list, axis=0)
    print(f"Latent features loaded from: {features_file_path}")
    return latent_features


# ============================================================================
# Adam Functions
# ============================================================================

def reconstruct_predict(prediction_file, coords_file, autoencoder_model, cluster_model, window_size, predictions_batch_size):
    """
    Reconstructs an image and its cluster predictions from a prediction file and coordinates file.
    
    Returns: reconstructed_img, cluster_img
    """
    # Load the windows for the image as a numpy file
    image_windows = np.load(prediction_file)
    # Load the image window coordinates
    image_windows_coordinates = dp.load_coordinates_file(coords_file)
    # Reconstruct the original image from the loaded image windows
    reconstructed_img = dp.reconstruct_image(image_windows,image_windows_coordinates,window_size)

    # Make a tensorflow data pipeline of just the image windows for this image.
    num_windows = image_windows.shape[0]
    print('\n---\nProcessing file {}'.format(os.path.basename(prediction_file)))

    # Predictions windows
    predict_dataset = create_tf_dataset_batched(
        [prediction_file], 
        batch_size=predictions_batch_size, 
        window_size=window_size,
        is_autoencoder=False, 
        shuffle=False)

    # make the latent features for each window using the autoencoder model 
    latent_predict_features, num_latent_predictions = extract_latent_features_to_disk_from_prebatched_windows(
        autoencoder_model, 
        predict_dataset, 
        '',                 # we are not saving these predictions to disk so don't need a folder or name
        features_name='',
        return_array=True,
        verbose=False)

    # make preductions 
    cluster_predictions = cluster_model.predict(latent_predict_features)

    # Build the reconstruction of the predicted cluster label data
    cluster_img = dp.reconstruct_cluster_image(image_windows_coordinates,window_size, cluster_predictions)

    # Pad the cluster image to the original image size
    cluster_img = ut.padded_cluster_img = ut.pad_cluster_image(reconstructed_img,cluster_img,window_size)
    
    return reconstructed_img, cluster_img

def remove_small_objects(ar, min_size=7,out=None):
    """
    Remove small objects from a binary array.
    
    Parameters:
    - ar: Input binary array.
    - min_size: Minimum size of objects to keep.
    
    
    Returns:
    - Binary array with small objects removed.
    """
    if out is None:
        out = ar.copy()
    else:
        out[:] = ar
    ccs = out
    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    
    
    return out

def remove_large_objects(ar, max_size=5000, out=None):
    """
    Remove large objects from a binary array.
    
    Parameters:
    - ar: Input binary array.
    - max_size: Maximum size of objects to keep.
    
    
    Returns:
    - Binary array with large objects removed.
    """
    if out is None:
        out = ar.copy()
    else:
        out[:] = ar
    ccs = out
    component_sizes = np.bincount(ccs.ravel())
    too_large = component_sizes > max_size
    too_large_mask = too_large[ccs]
    out[too_large_mask] = 0
    
    return out

def detect_features_find_centres(cluster_img, max_size=20000, area_threshold=10,):
    data_postprocess = cluster_img.astype("int")  # Convert to boolean type for morphological operations

    #remove background
    large_removed = remove_large_objects(data_postprocess, max_size=max_size)
    #small_removed = remove_small_objects(large_removed, min_size=1000)  # Remove small objects with a minimum size of 100 pixels


    area_closed = morphology.area_closing(large_removed, area_threshold=area_threshold)  # Apply closing operation to fill small holes
    area_opened = morphology.area_opening(area_closed, area_threshold=area_threshold)  # Apply opening operation to remove small objects
    #eroded = morphology.erosion(area_opened, morphology.disk(4))  # Apply opening operation to remove small objects


    foreground = area_opened > 0  # Convert to boolean type for morphological operations
    #Perform connected component labeling
    #labeled_array, num_features = ski.measure.label(foreground, return_num=True, connectivity=2)  # Use connectivity=2 for 8-connectivity

    labeled_array, num_features = scipy.ndimage.label(foreground)  # Use a 3x3 structure for connectivity

    #connect_labels = ski.measure.label(labeled_array, connectivity=2)  # Use connectivity=2 for 8-connectivity
    centers = scipy.ndimage.center_of_mass(foreground, labeled_array, range(1, num_features + 1))
    centers = np.array(centers)  # Convert to numpy array for easier manipulation

    #centers = [list(region.centroid) for region in regions]
    #centers = np.array(centers)  # Transpose to get x and y coordinates

    #features = large_removed
    features = labeled_array
    return features, centers, labeled_array, num_features

def detect_features_better(cluster_img, max_size=3000, area_threshold=10):
    
    data = cluster_img.astype("int32")
    boundaries = segmentation.find_boundaries(data, mode="outer")

    # Create a copy to work with
    boundary_image = boundaries.astype(int)

    # Label connected components (regions between boundaries)
    labeled_regions, num_regions = scipy.ndimage.label(~boundaries)

    # Calculate area of each region
    areas = {}
    for region_id in range(1, num_regions + 1):
        area = np.sum(labeled_regions == region_id)
        areas[region_id] = area
        
    large_removed = remove_large_objects(labeled_regions, max_size=max_size)

    area_closed = morphology.area_closing(large_removed, area_threshold=area_threshold)  # Apply closing operation to fill small holes
    area_opened = morphology.area_opening(area_closed, area_threshold=area_threshold)  # Apply opening operation to remove small objects

    foreground = area_opened > 0  # Convert to boolean type for morphological operations
    dilated = morphology.binary_dilation(foreground, morphology.disk(1))  # Dilate the foreground to connect nearby features
    closed = morphology.binary_closing(dilated, morphology.disk(2))  # Close small gaps in the foreground
    labeled_array, num_features = scipy.ndimage.label(closed)  # Use a 3x3 structure for connectivity

    #clearning the background
    closed = morphology.closing(labeled_array, morphology.disk(2))  # Close small gaps in the foreground
    opened = morphology.opening(closed, morphology.disk(2))  # Open small gaps in the foreground
    
    centers = scipy.ndimage.center_of_mass(opened, labeled_array, range(1, num_features + 1))
    centers = np.array(centers)  # Convert to numpy array for easier manipulation

    
    return labeled_array, centers, num_features

def display_reconstructed_and_cluster_images_and_extracted_features(reconstructed_img, cluster_img, features_img, centers, 
                                             save_to_disk=False, output_path=None, image_name='img', dpi=150):
    """
    Display side-by-side images: the reconstructed input image, the cluster labels,
    and the highlighted features.

    Parameters:
    - reconstructed_img (ndarray): The reconstructed image, typically the output
      from an autoencoder or similar model.
    - cluster_img (ndarray): The image showing cluster labels, usually resulting
      from clustering analysis on the latent space.
    - features_img (ndarray): The image showing highlighted features, such as
      detected regions or points of interest.
    - centers (list): List of coordinates for the centers of detected features.
    #- show_overlay (bool): Optional; if True, displays a third panel with an overlay
    #  of the reconstructed image and cluster labels. Default is True.
    - save_to_disk (bool): Optional; if True, saves the image to disk instead of displaying it. Default is False.
    - output_path (str): The path to save the image if save_to_disk is True. Required if save_to_disk is True.
    - dpi (int): Optional; the resolution in dots per inch for saving the image. Default is 300.

    Returns:
    - None: This function either displays the plot or saves it to disk based on save_to_disk.
    """
    # Output directory with timestamp
    if save_to_disk:
      if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory created: {output_path}")
        
    n_plots = 3  # Number of plots to display
    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    
    # Plot the first image (Reconstructed Image)
    ax[0].imshow(reconstructed_img, cmap='viridis')
    ax[0].set_title('Input Image')
    ax[0].axis('off')  # Remove axis labels

    # Plot the second image (Cluster Labels)
    ax[1].imshow(cluster_img, cmap='viridis', interpolation='nearest')
    ax[1].set_title('Cluster Labels')
    ax[1].axis('off')  # Remove axis labels
    
    ax[2].imshow(features_img, cmap='viridis', interpolation='nearest')
    
    ax[2].set_title('Detected Features')
    ax[2].axis('off')  # Remove axis labels
    
    if len(centers) > 0:
      ax[0].scatter(centers[:, 1], centers[:, 0], color='red', s=5, alpha=0.5)  # Highlight the centers of detected features
      ax[1].scatter(centers[:, 1], centers[:, 0], color='red', s=5)  # Highlight the centers of detected features
      ax[2].scatter(centers[:, 1], centers[:, 0], color='red', s=5)  # Highlight the centers of detected features
        
    plt.tight_layout()
    
    # Save to disk or display
    if save_to_disk:
        if output_path:
            output_path = os.path.join(output_path, f"{image_name}.jpg")
            plt.savefig(output_path, format='jpg', dpi=dpi)
            print(f"Image saved to {output_path} with dpi={dpi}")
        else:
            print("Error: output_path must be specified when save_to_disk is True.")
    else:
        plt.show()

    # Close the figure to release memory
    plt.close(fig)


def extract_feature_windows(image, centers, px=128):

    
    #print('EXTRRACTING WINDOWS. ORIG IMAGE: Height {} Width {}'.format(height, width))
    height, width = image.shape


    # Extract windows and record coordinates
    windows = np.zeros((len(centers), px * 2, px * 2), dtype=image.dtype)  # Initialize windows array
    centers = np.round(centers).astype(int)  # Ensure centers are integers for indexing
    px = int(px)  # Ensure px is an integer for indexing
    for i in range(len(centers)):
        y = centers[i, 0]
        x = centers[i, 1]
            
        # Adjust start positions to ensure full windows at the edges
        if y + px > height:
            y = height - px
        if x + px > width:
            x = width - px
        if y - px < 0:
            y = px
        if x - px < 0:
            x = px

        # Extract windows with consistent size
        window = image[y-px:y+px, x-px:x+px]
        windows[i, :, :] = window

        # if window.shape[0] != 32 or window.shape[1] != 32:
        #     print('\n\n*************\n window number {}: {} {}\n\n'.format(window_number, window.shape[0],window.shape[1]))
            
        #     plt.imshow(window)
        #     plt.show()

    return np.array(windows)
#======================================================================
#clustering feature windows
#======================================================================

def extract_latent_features_to_disk_from_prebatched_feature_windows(
    autoencoder_model,
    dataset,
    features_path,
    bottleneck_layer_name='bottleneck',
    features_name='latent_features',
    return_array=False,
    verbose=False):
    """
    Extracts latent features from a pre-batched TensorFlow dataset, where each batch is processed
    and saved directly as a corresponding batch of latent features.

    Parameters:
    - autoencoder_model: The trained autoencoder model to extract latent features.
    - dataset: The TensorFlow dataset containing pre-batched input windows.
    - features_path: The directory where latent features should be saved.
    - bottleneck_layer_name: The name of the bottleneck layer in the autoencoder model.
    - features_name: The base name for the saved feature files.
    - verbose: Whether to print progress information.

    Structure of Saved Data:
    - Each batch is saved as a .npy file containing a 2D NumPy array.
    - Shape: (batch_size, latent_dim), where:
        - batch_size is the number of samples in the batch.
        - latent_dim is the flattened size of the bottleneck layer.
    - Each row represents the latent feature vector for a single input window.
    """
    # Create encoder model to extract latent space from the specified bottleneck layer
    encoder_model = tf.keras.Model(
        inputs=autoencoder_model.input,
        outputs=autoencoder_model.get_layer(bottleneck_layer_name).output
    )

    if return_array:
        latent_features_all = []
    else:
        # Ensure the output directory exists
        if not os.path.exists(features_path):
            os.makedirs(features_path)

    # Process each batch in the dataset
    for i, batch in enumerate(dataset):
        inputs = batch if isinstance(batch, tf.Tensor) else batch[0]  # Handle datasets with or without labels
        inputs_resized = tf.image.resize(inputs, (32,32))  # Resize inputs to match the model input size
        batch_shape = tf.shape(inputs_resized).numpy()
        if verbose: 
            print(f"Processing batch {i + 1}, input shape: {batch_shape}")
        
        # Compute latent features for the current batch
        latent_features_batch = encoder_model.predict(inputs_resized, verbose=0)
        
        # Flatten latent features if necessary
        if len(latent_features_batch.shape) > 2:
            flattened_latent_features = tf.reshape(
                latent_features_batch, 
                [latent_features_batch.shape[0], -1]
            ).numpy()
        else:
            flattened_latent_features = latent_features_batch
        
        # Validate output shape (N, M)
        latent_shape = flattened_latent_features.shape

        if i==0: 
            sample_batch_shape = flattened_latent_features.shape

        if verbose:
            print(f"Latent features shape for batch {i + 1}: {latent_shape}")
        else:
            if i>0 and i%100==0:
                print(i,)
            else:
                print('.',end='')
        
        if return_array:
            latent_features_all.append(flattened_latent_features)
        else:
            # Save the latent features batch to disk
            file_path = os.path.join(features_path, f"{features_name}_batch_{i}.npy")
            np.save(file_path, flattened_latent_features)
            if verbose:
              print(f"Saved latent features for batch {i + 1} to {file_path}")
    
    if return_array:
        # Combine all latent features into a single array
        latent_features_all = np.concatenate(latent_features_all, axis=0)
        print(f"Combined latent features shape: {latent_features_all.shape}")
        return latent_features_all, latent_features_all.shape[0]
    else:
        print(f"\nAll latent features have been saved to {features_path}.")
        print(f"Sample batch shape: {sample_batch_shape}")


def train_dbscan(data_pipeline, eps=0.5, min_samples=5):
    
    
    dbscan = sklearn.cluster.DBSCAN(
        eps=eps,  # Maximum distance between two samples for one to be considered
        min_samples=min_samples,  # Minimum number of samples in a neighborhood for a point to
        # be considered a core point
        metric='euclidean',  # Distance metric to use
    )
    
    


    for i, batch in enumerate(data_pipeline):
        # Convert batch to NumPy array if needed
        latent_features_batch = np.array(batch) if isinstance(batch, tf.Tensor) else batch
        latent_features_batch = latent_features_batch.reshape(-1, latent_features_batch.shape[-1])

        # Perform fitting with the current batch
        dbscan.fit(latent_features_batch)


        print(f"Batch {i+1} processed")
        sys.stdout.flush()

    return dbscan


def train_spectral_clustering(data_pipeline, n_clusters=10, affinity='nearest_neighbors', n_neighbors=10):
    """
    Trains a Spectral Clustering model using latent feature data from a TensorFlow data pipeline.

    Parameters:
    -----------
    data_pipeline : tf.data.Dataset
        A TensorFlow Dataset object yielding batches of latent feature vectors.
    n_clusters : int, optional
        The number of clusters to form. Default is 10.
    affinity : str, optional
        The type of affinity to use. Default is 'nearest_neighbors'.
    n_neighbors : int, optional
        Number of neighbors to use for the nearest neighbors graph. Default is 10.

    Returns:
    --------
    SpectralClustering
        A trained Spectral Clustering model fitted to the latent features from the dataset.
    """
    
    # Initialize Spectral Clustering model
    spectral_clustering = sklearn.cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans',  # Use k-means for label assignment
        random_state=42  # For reproducibility
    )
    
    all_latent_features = gather_latent_features(data_pipeline)

    # Fit the model on the entire dataset at once
    spectral_clustering.fit(all_latent_features)

    return spectral_clustering
    
#==================================
# W-net
#==================================

def build_unet(input_tensor, name_prefix='unet'):
    tf.random.set_seed(42)  # Set seed for reproducibility
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_conv1')(input_tensor)
    d1 = layers.Dropout(0.1, name=f'{name_prefix}_drop1')(c1)
    p1 = layers.MaxPooling2D(2, name=f'{name_prefix}_pool1')(d1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_conv2')(p1)
    d2 = layers.Dropout(0.1, name=f'{name_prefix}_drop2')(c2)
    p2 = layers.MaxPooling2D(2, name=f'{name_prefix}_pool2')(d2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_conv3')(p2)
    d3 = layers.Dropout(0.1, name=f'{name_prefix}_drop3')(c3)
    p3 = layers.MaxPooling2D(2, name=f'{name_prefix}_pool3')(d3)
    # Bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_bottleneck')(p3)

    # Decoder
    u1 = layers.UpSampling2D(size=2, name=f'{name_prefix}_up1')(b)
    u1 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name=f'{name_prefix}_upconv1')(u1)
    u1 = layers.concatenate([u1, c3], axis=-1, name=f'{name_prefix}_skip1')
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name=f'{name_prefix}_conv4')(u1)
    
    u2 = layers.UpSampling2D(size=2, name=f'{name_prefix}_up2')(c5)
    u2 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name=f'{name_prefix}_upconv2')(u2)
    u2 = layers.concatenate([u2, c2], axis=-1, name=f'{name_prefix}_skip2')
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name=f'{name_prefix}_conv5')(u2)
    
    u3 = layers.UpSampling2D(size=2, name=f'{name_prefix}_up3')(c6)
    u3 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name=f'{name_prefix}_upconv3')(u3)
    u3 = layers.concatenate([u3, c1], axis=-1, name=f'{name_prefix}_skip3')
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal', name=f'{name_prefix}_conv6')(u3)
    
    return c7


def build_wnet(window_size, num_classes=5):
    inputs = tf.keras.Input(shape=(window_size, window_size, 1), name='input_image')

    # First U-Net: segmentation
    seg_features = build_unet(inputs, name_prefix='seg')
    seg_output = layers.Conv2D(num_classes, 1, activation='softmax', name='seg_output')(seg_features)

    # Second U-Net: reconstruction from segmentation
    recon_features = build_unet(seg_output, name_prefix='recon')
    recon_output = layers.Conv2D(num_classes, 1, activation='softmax', name='recon_output')(recon_features)

    return tf.keras.Model(inputs=inputs, outputs=[seg_output, recon_output], name='wnet')