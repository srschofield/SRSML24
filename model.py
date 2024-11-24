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

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib # for saving cluster model

# ============================================================================
# System information
# ============================================================================

def print_system_info():
    """
    Prints Python, TensorFlow, and system information, and checks if TensorFlow is using the GPU with CUDA.
    Also prints CPU and relevant GPU information.
    """
    # Print Python version
    print(f"Python version: {sys.version}")
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Print system information
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if TensorFlow is using GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Number of GPUs available to TensorFlow: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU Device: {gpu}")
        print("CUDA is being used for TensorFlow.")
    else:
        print("No GPU available, TensorFlow running on CPU")
    
    # Check TensorFlow build information
    print(f"TensorFlow is built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"TensorFlow is built with ROCm: {tf.test.is_built_with_rocm()}")

    # Print summary of GPU availability
    if gpus:
        print("\n>>> Running with GPU available <<<  ({})\n".format(platform.platform()))
    else:
        print("\n@@@ NO GPU @@@\n")


    # Call the functions to get detailed CPU and GPU information
    #get_detailed_cpu_info()
    #get_detailed_gpu_info()



# FOR LINUX
def get_detailed_cpu_info():
    """
    Retrieves detailed CPU information using lscpu on Linux.
    """
    try:
        # Get detailed CPU info using lscpu
        cpu_info = subprocess.check_output(["lscpu"]).decode("utf-8")
        print("\nDetailed CPU Information:")
        print(cpu_info)
    except Exception as e:
        print(f"Could not retrieve detailed CPU info: {e}")



def get_detailed_gpu_info():
    """
    Retrieves detailed GPU information using nvidia-smi on Linux.
    """
    try:
        # Get GPU information using nvidia-smi
        gpu_info = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print("\nDetailed GPU Information:")
        print(gpu_info)
    except Exception as e:
        print(f"Could not retrieve detailed GPU info: {e}")

# FOR MAC
# def get_detailed_cpu_info():
#     """
#     Retrieves detailed CPU information from macOS using sysctl.
#     """
#     try:
#         # Get CPU brand and core count using sysctl
#         cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
#         cpu_cores = subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).decode("utf-8").strip()
        
#         print("\nDetailed CPU Information:")
#         print(f"CPU Model: {cpu_brand}")
#         print(f"Total Number of Cores: {cpu_cores}")
        
#     except Exception as e:
#         print(f"Could not retrieve detailed CPU info: {e}")

# def get_detailed_gpu_info():
#     """
#     Retrieves detailed GPU information from macOS system_profiler, excluding display info.
#     """
#     try:
#         # Run system_profiler and decode the output
#         gpu_info = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode("utf-8")

#         # Filter out lines related to "Display" and only include lines with "Chipset Model", "Core", and "Metal"
#         filtered_gpu_info = []
#         include_info = False
#         for line in gpu_info.splitlines():
#             line = line.strip()
#             if "Chipset Model" in line or "Total Number of Cores" in line or "Metal" in line:
#                 include_info = True  # Start including relevant GPU info
#             if "Displays:" in line:  # Stop when display information starts
#                 include_info = False
#             if include_info:
#                 filtered_gpu_info.append(line)

#         # Print the filtered GPU information
#         print("\nRelevant GPU Information (from system profiler):")
#         for info in filtered_gpu_info:
#             print(info)

    except Exception as e:
        print(f"Could not retrieve detailed GPU info: {e}")


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

def save_model(model, model_path, model_name='autoencoder'):
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

    # Full model file path with .h5 extension
    model_file_path = os.path.join(model_path, model_name + '.keras')

    # Save the model
    model.save(model_file_path)
    print(f"Model saved at: {model_file_path}")


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



def save_history(history, model_path, model_name='autoencoder', 
                 loss_name='loss', val_loss_name='val_loss', 
                 metrics=None, val_metrics=None):
    """
    Saves the training history to a text file in the same directory as the model.
    
    Parameters:
    - history: The history object returned by the `model.fit()` method.
    - model_path: The directory where the model and history will be saved.
    - model_name: The name of the history file (without extension). Default is 'autoencoder'.
    - loss_name: The name of the training loss metric in the history. Default is 'loss'.
    - val_loss_name: The name of the validation loss metric in the history. Default is 'val_loss'.
    - metrics: A list of names of training metrics to be included in the history. Default is None.
    - val_metrics: A list of names of validation metrics to be included in the history. Default is None.
    """
    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Full history file path with .txt extension
    history_path = os.path.join(model_path, model_name + '_history.txt')

    # Extract history data
    epochs = range(1, len(history.history[loss_name]) + 1)
    train_loss = history.history[loss_name]
    val_loss = history.history.get(val_loss_name, [])

    # Open the file for writing
    with open(history_path, 'w') as f:
        # Write headers (loss and optional metrics)
        headers = f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}"
        
        # If metrics are provided, add them to the headers
        if metrics:
            for metric in metrics:
                headers += f"{'Train ' + metric:<15}"
        if val_metrics:
            for val_metric in val_metrics:
                headers += f"{'Val ' + val_metric:<15}"
        f.write(headers + "\n")

        # Write values for each epoch
        for epoch in epochs:
            # Basic loss values
            line = f"{epoch:<10}{train_loss[epoch-1]:<15.5f}{val_loss[epoch-1] if val_loss else '':<15.5f}"
            
            # Add train and validation metrics, if provided
            if metrics:
                for metric in metrics:
                    line += f"{history.history[metric][epoch-1]:<15.5f}"
            if val_metrics:
                for val_metric in val_metrics:
                    line += f"{history.history[val_metric][epoch-1]:<15.5f}"
            
            f.write(line + "\n")

    print(f"History saved to {history_path}")


# ============================================================================
# Model - UNET
# ============================================================================


def build_autoencoder(window_size, model_name='autoencoder'):
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Define the input shape
    inputs = tf.keras.Input(shape=(window_size, window_size, 1), name='Input_windows')
    
    # Encoder (Contracting Path)
    c1 = layers.Conv2D(name='Convolutional_1',
                       filters=32, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(inputs)
    
    d1 = layers.Dropout(0.1, name='Dropout_1')(c1)
    
    p1 = layers.MaxPooling2D(name='MaxPooling_1', pool_size=2)(d1)  # (window_size / 2, window_size / 2, 32)
    
    c2 = layers.Conv2D(name='Convolutional_2',
                       filters=64, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(p1)
    
    d2 = layers.Dropout(0.1, name='Dropout_2')(c2)
    
    p2 = layers.MaxPooling2D(name='MaxPooling_2', pool_size=2)(d2)  # (window_size / 4, window_size / 4, 64)
    
    c3 = layers.Conv2D(name='Convolutional_3',
                       filters=128, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(p2)
    
    d3 = layers.Dropout(0.1, name='Dropout_3')(c3)
    
    p3 = layers.MaxPooling2D(name='MaxPooling_3', pool_size=2)(d3)  # (window_size / 8, window_size / 8, 128)
    
    # Bottleneck (lowest point in the U)
    c4 = layers.Conv2D(name='Bottleneck',
                       filters=256, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(p3)
    
    # Decoder (Expansive Path)
    u1 = layers.UpSampling2D(name='UpSampling_1', size=2)(c4)  # (window_size / 4, window_size / 4, 128)
    u1 = layers.Conv2D(name='UpConv_1',
                       filters=128, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(u1)
    
    # Skip connection with c3
    u1 = layers.concatenate([u1, c3], axis=-1, name='Skip_Connection_1')  # Match c3's dimensions
    
    c5 = layers.Conv2D(name='Convolutional_4',
                       filters=128, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(u1)
    
    u2 = layers.UpSampling2D(name='UpSampling_2', size=2)(c5)  # (window_size / 2, window_size / 2, 64)
    u2 = layers.Conv2D(name='UpConv_2',
                       filters=64, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(u2)
    
    # Skip connection with c2
    u2 = layers.concatenate([u2, c2], axis=-1, name='Skip_Connection_2')
    
    c6 = layers.Conv2D(name='Convolutional_5',
                       filters=64, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(u2)
    
    u3 = layers.UpSampling2D(name='UpSampling_3', size=2)(c6)  # (window_size, window_size, 32)
    u3 = layers.Conv2D(name='UpConv_3',
                       filters=32, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(u3)
    
    # Skip connection with c1
    u3 = layers.concatenate([u3, c1], axis=-1, name='Skip_Connection_3')
    
    c7 = layers.Conv2D(name='Convolutional_6',
                       filters=32, 
                       kernel_size=3, 
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(u3)
    
    outputs = layers.Conv2D(name='Output',
                            filters=1, 
                            kernel_size=1, 
                            activation='sigmoid',  # Change to 'linear' if the task requires it
                            padding='same')(c7)
    
    # Define the U-Net model with the specified name
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    return model



# ============================================================================
# Model training information
# ============================================================================

def plot_training_history(history, 
                          loss_name='loss', 
                          val_loss_name='val_loss', 
                          metric_names=['mse', 'mae'], 
                          val_metric_names=['val_mse', 'val_mae'],
                          save_to_disk=False,
                          output_path=None,
                          dpi=150):
    """
    Plots the training and validation loss and metrics from the history object returned by model.fit().
    
    Parameters:
    history: History object
        The history object returned by the model.fit() function.
    loss_name: str, optional
        The name of the loss key in the history object. Default is 'loss'.
    val_loss_name: str, optional
        The name of the validation loss key in the history object. Default is 'val_loss'.
    metric_names: list of str, optional
        The names of the training metric keys in the history object. Default is ['mse', 'mae'].
    val_metric_names: list of str, optional
        The names of the validation metric keys in the history object. Default is ['val_mse', 'val_mae'].
    save_to_disk: bool, optional
        If True, saves the plot to disk instead of displaying it on screen. Default is False.
    output_path: str, optional
        The path to save the plot if save_to_disk is True. Must be specified if save_to_disk is True.
    """
    # Extract values from history
    loss = history.history.get(loss_name, None)
    val_loss = history.history.get(val_loss_name, None)
    
    # Extract metrics
    metrics = {metric: history.history.get(metric, None) for metric in metric_names}
    val_metrics = {val_metric: history.history.get(val_metric, None) for val_metric in val_metric_names}
    
    epochs = range(1, len(loss) + 1) if loss else []
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    if loss and val_loss:
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    else:
        print("Loss data is not available in the provided history object.")
    
    # Plot Metrics
    if metrics and val_metrics:
        plt.subplot(1, 2, 2)
        for metric, val_metric in zip(metric_names, val_metric_names):
            if metrics[metric] is not None and val_metrics[val_metric] is not None:
                plt.plot(epochs, metrics[metric], label=f'Training {metric}')
                plt.plot(epochs, val_metrics[val_metric], label=f'Validation {val_metric}', linestyle='dashed')
        plt.title('Metrics over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.legend()
    else:
        print("Metric data is not available in the provided history object.")
    
    plt.tight_layout()
    
    # Display or save the plot
    if save_to_disk:
        if output_path:
            output_path = output_path + '/history.jpg'
            plt.savefig(output_path, format='jpg', dpi=dpi)
            print(f"Plot saved to {output_path}")
        else:
            print("Error: output_path must be specified when save_to_disk is True.")
    else:
        plt.show()



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
        ax[2].imshow(reconstructed_img, cmap='viridis')
        ax[2].imshow(cluster_img, cmap='turbo', alpha=0.6)  # Overlay with transparency
        ax[2].set_title('Overlayed Image with Cluster Labels')
        ax[2].axis('off')  # Remove axis labels

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
    features_name='latent_features'
):
    """
    Extracts latent features from a pre-batched TensorFlow dataset, where each batch is processed
    and saved directly as a corresponding batch of latent features.

    Parameters:
    - autoencoder_model: The trained autoencoder model to extract latent features.
    - dataset: The TensorFlow dataset containing pre-batched input windows.
    - features_path: The directory where latent features should be saved.
    - features_name: The base name for the saved feature files.

    Structure of Saved Data:
    - Each batch is saved as a .npy file containing a 2D NumPy array.
    - Shape: (batch_size, latent_dim), where:
        - batch_size is the number of samples in the batch.
        - latent_dim is the dimensionality of the bottleneck layer in the autoencoder.
    - Each row represents the latent feature vector for a single input window.
    """
    # Create encoder model to extract latent space from the 'Bottleneck' layer
    encoder_model = tf.keras.Model(
        inputs=autoencoder_model.input,
        outputs=autoencoder_model.get_layer('Bottleneck').output
    )
    
    # Ensure the output directory exists
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    # Process each pre-batched input and save latent features directly
    for i, batch in enumerate(dataset):
        inputs = batch[0]  # Extract inputs, ignoring labels if present
        batch_shape = inputs.shape
        print(f"Processing batch {i + 1}, shape: {batch_shape}")
        
        # Compute latent features for the current batch
        latent_features_batch = encoder_model.predict(inputs, verbose=0)
        
        # Save the latent features batch to disk
        file_path = os.path.join(features_path, f"{features_name}_batch_{i}.npy")
        np.save(file_path, latent_features_batch)
        print(f"Saved latent features for batch {i + 1} to {file_path}")

    print(f"\nAll latent features have been saved to {features_path}.")


def extract_latent_features_to_array(autoencoder_model, input_data):
    """
    Extracts latent features from an individual input (or batch of inputs) and returns them as a NumPy array.

    Parameters:
    - autoencoder_model: The trained autoencoder model to extract latent features.
    - input_data: A single input or a batch of inputs for which to extract latent features.

    Returns:
    - np.ndarray: A 2D NumPy array containing the latent feature vectors.
                  Shape: (num_samples, latent_dim), where num_samples is the number of samples in input_data.
    """
    # Create encoder model to extract latent space from 'Bottleneck' layer
    encoder_model = tf.keras.Model(
        inputs=autoencoder_model.input,
        outputs=autoencoder_model.get_layer('Bottleneck').output
    )
    
    # Predict latent features for the input data
    latent_features = encoder_model.predict(input_data, verbose=0)
    
    # Flatten latent features to 2D if necessary
    latent_features_flat = latent_features.reshape(latent_features.shape[0], -1)
    
    num_features = len(latent_features_flat)
    return latent_features_flat, num_features


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


def create_latent_features_tf_dataset_from_array(latent_features_array, batch_size=32):
    """
    Creates a TensorFlow data pipeline from a NumPy array of latent features, preparing them
    for clustering or further analysis in batches.

    Parameters:
    -----------
    latent_features_array : np.ndarray
        A NumPy array containing latent features, with shape (N, M), where N is the number of samples,
        and M is the dimensionality of the latent features.
    batch_size : int, optional
        The number of latent feature vectors to include in each batch. Default is 32.

    Returns:
    --------
    tf.data.Dataset
        A TensorFlow Dataset object that yields batches of latent features, each with shape
        (batch_size, M), where M is the dimensionality of the features. The dataset is configured to
        prefetch batches for optimized loading performance.
    """
    # Ensure the input is a NumPy array
    if not isinstance(latent_features_array, np.ndarray):
        raise ValueError("latent_features_array must be a NumPy array")

    # Convert the NumPy array to a TensorFlow dataset
    data_pipeline = tf.data.Dataset.from_tensor_slices(latent_features_array)
    
    # Batch and prefetch the data
    data_pipeline = data_pipeline.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Print details about the created pipeline
    print(f"Data pipeline created from NumPy array with shape {latent_features_array.shape}, batch size: {batch_size}")
   
    for batch in data_pipeline.take(1):  # Preview the first batch
        print(f"Batch shape: {batch.shape}")
    
    return data_pipeline


def train_kmeans(data_pipeline, batch_size=2048, num_clusters=10, n_init=10, max_iter=300, reassignment_ratio=0.01):
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
    initial_kmeans = KMeans(n_clusters=num_clusters, init='k-means++').fit(subset)
    
    # Initialize MiniBatchKMeans with precomputed centroids
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

    print(f"Final inertia after training: {kmeans.inertia_}")
    return kmeans, convergence_history



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
