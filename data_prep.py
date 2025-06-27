#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation for ML analysis of STM MTRX data

This file contains functions for preparing data
for training and prediction.
    
@author: Steven R. Schofield 

Created October 2024

"""

# ============================================================================
# Module dependencies
# ============================================================================

import os, sys

# Third-party library imports
import numpy as np  # Fundamental package for array computing with Python, useful for numerical operations on large, multi-dimensional arrays and matrices
import spiepy  # Custom or third-party library for specific image processing functions, such as flattening images
import gc # garbage collector for memory management

from PIL import Image  # Python Imaging Library (PIL) is used for opening, manipulating, and saving many different image file formats

# # Custom module import
import access2thematrix  # Custom module for handling MTRX file formats, presumably specific to a particular imaging or data format

# # Instantiating the MtrxData class for reading MTRX files
# # This object provides methods to open and interact with MTRX data files
mtrx = access2thematrix.MtrxData()

import cv2

import random

import matplotlib.pyplot as plt

from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

from scipy.signal import savgol_filter

# ============================================================================
# Google Colab
# ============================================================================
def colab_init():
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    else:
        print("Not running on Google Colab")
        return False

def convert_path_to_colab(local_path):
    # Split the local path by '/' and get the relevant parts
    path_parts = local_path.split('/')
    
    # Check if the path contains 'My Drive', which is the key part we need to convert
    if 'My Drive' in path_parts:
        # Find the index where 'My Drive' occurs
        my_drive_index = path_parts.index('My Drive')
        
        # Keep the relevant part of the path from 'My Drive' onwards
        sub_path = '/'.join(path_parts[my_drive_index + 1:])
        
        # Construct the Colab path
        colab_path = f'/content/drive/MyDrive/{sub_path}'
        
        return colab_path
    else:
        raise ValueError("The provided path does not contain 'My Drive'.")

# def install_modules_on_colab():
#     try:
#         import spiepy
#         print("Spiepy is installed.")
#     except ModuleNotFoundError:
#         print("Spiepy is installed not installed. Installing now...")
#         !pip install spiepy
#         import spiepy

#     # The 'access2thematrix' package is a python library for loading Scienta Omicron
#     # data
#     try:
#         import access2thematrix
#         print("access2thematrix is installed.")
#     except ModuleNotFoundError:
#         print("access2thematrix is installed not installed. Installing now...")
#         !pip install access2thematrix
#         import access2thematrix

# ============================================================================
# Misc
# ============================================================================

def current_datetime():
    """
    Function: get_current_datetime
    Description: This function retrieves the current date and time, formats it as a string, prints it, 
                 and returns the current date and time object.
    Returns: A datetime object representing the current date and time.
    """
    current_datetime = datetime.now()
    formatted_date_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print('Current time {}'.format(formatted_date_time))
    return current_datetime

def elapsed_time(start, end):
    """
    Function: elapsed_time
    Description: This function takes two datetime objects (start and end), calculates the difference, 
                 and returns a formatted string representing the elapsed time in hours, minutes, seconds, and microseconds.
    Parameters:
    - start: The start datetime object.
    - end: The end datetime object.
    Returns: A string representing the elapsed time.
    """
    delta = end - start
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{hours} hours, {minutes} minutes, {seconds} seconds."
    print('Elapsed time: {}'.format(formatted_time))
    return 


# ============================================================================
# Process MTRX
# ============================================================================

def cleanup(*vars):
    for var in vars:
        try:
            del var
        except NameError:
            pass  # Ignore variables that were never created
    gc.collect()


def process_mtrx_files(mtrx_paths, save_data_path, **kwargs):
    flatten_method = kwargs.get('flatten_method', 'iterate_mask')
    pixel_density = kwargs.get('pixel_density', 15.0)
    pixel_ratio = kwargs.get('pixel_ratio', 0.5)
    data_scaling = kwargs.get('data_scaling', 2e9)
    window_size = kwargs.get('window_size', 32)
    window_pitch = kwargs.get('window_pitch', 32)
    save_windows = kwargs.get('save_windows', True)
    save_jpg = kwargs.get('save_jpg', False)
    verbose = kwargs.get('verbose', False)
    together = kwargs.get('together', False)
    collate = kwargs.get('collate', False)
    resample = kwargs.get('resample', True)
    save_meta = kwargs.get('save_meta', True)
    cmap = kwargs.get('cmap','gray')
    save_window_jpgs = kwargs.get('save_window_jpgs', False)

    save_windows = bool(save_windows) # convert to boolean
    # save_raw = bool(save_raw) # convert to boolean
    save_jpg = bool(save_jpg) # convert to boolean
    verbose = bool(verbose)  # convert to boolean
    together = bool(together)    # convert to boolean
    collate = bool(collate)    # convert to boolean
    resample = bool(resample)    # convert to boolean
    save_meta = bool(save_meta)
    save_window_jpgs = bool(save_window_jpgs)

    jpg_path = os.path.join(save_data_path,'jpg')
    windows_path = os.path.join(save_data_path,'windows')
    windows_jpg_path = os.path.join(save_data_path,'windows-jpg')

    # If only a single filename is given, convert this to list
    if isinstance(mtrx_paths, str):
        mtrx_paths = [mtrx_paths]

    total_files = len(mtrx_paths)
    print(f'There are {total_files} files to process')

    # Dictionary for the scan directions
    scan_direction_map = {
        "forward/up": 'FU',
        "backward/up": 'BU',
        "forward/down": 'FD',
        "backward/down": 'BD'
    }

    for idx, mtrx_path in enumerate(mtrx_paths, start=1):

        if verbose:
            print('___________________________________________')
            print(f'Processing file {idx} of {total_files}')
            print(f'File: {mtrx_path}')

        # # Output to screen to indicate progress
        if not verbose:
            if idx % 100 == 0:
                print(' {}'.format(idx),end='\n')
            else:
                print('.', end='')
                
        # LOAD MTRX FILE
        imgAll, _ = load_mtrx_data(mtrx_path)

        if imgAll is None:
            continue

        file_name_without_ext = os.path.splitext(os.path.basename(mtrx_path))[0]

        for scanDirection in imgAll:
            try:
                MTRXimg, _ = mtrx.select_image(imgAll[scanDirection])
                scan_direction_full = imgAll[scanDirection]  # Get the full scan direction name
                scan_direction = scan_direction_map[scan_direction_full]
            except Exception as e:
                print(f"Error processing scan direction {scanDirection} for {file_name_without_ext}: {e}")
                continue

            # Get the image data
            img = MTRXimg.data

            # Skip if the image is smaller than the size of a single window in any dimension
            if img.shape[0] <= window_size or img.shape[1] <= window_size:
                continue

            # Extract additional parameters
            additional_params = extract_regulation_parameters_from_mtrx_data(mtrx)

            # Determine if the scan direction is forward or backward
            if 'forward' in scan_direction_full.lower():
              selected_params = {
                    'bias': additional_params.get('bias_forward'),
                    'bias_unit': additional_params.get('bias_forward_unit'),
                    'current': additional_params.get('current_forward'),
                    'current_unit': additional_params.get('current_forward_unit')
              }
            else:  # backward
              selected_params = {
                    'bias': additional_params.get('bias_backward'),
                    'bias_unit': additional_params.get('bias_backward_unit'),
                    'current': additional_params.get('current_backward'),
                    'current_unit': additional_params.get('current_backward_unit')
              }
            # Create a metadata dictionary with width, height, angle, scan direction, and selected parameters
            metadata = {
                'filename': file_name_without_ext,
                'width': int(MTRXimg.width * 1e9),
                'height': int(MTRXimg.height * 1e9),
                'angle': MTRXimg.angle,
                'scan_direction': scan_direction_full,  # Add the scan direction to metadata
                **selected_params  # Add the selected forward or backward parameters
            }

            cleanup(MTRXimg)
          
            py, px = img.shape

            if verbose:
                #print(f"Width {metadata['width']} nm. Height {metadata['height']} nm. Angle {metadata['angle']} degrees.  Original px {px}, py {py}.")            
                print(f"[{idx}] [{scan_direction_full}] Image Metadata: ({px}x{py}). Width = {metadata['width']} nm, Height = {metadata['height']} nm, Angle = {metadata['angle']}°.")

            # Skip the file if the original pixel numbers are fewer than the window size in either dimension
            if px < window_size or py < window_size:
                if verbose:
                    print(f"[{idx}] [{scan_direction_full}] Skipping image with size ({px}x{py}) as it is smaller than the window size ({window_size}x{window_size}).")
                    #print("{} Skipping image of size ({}x{}) because it is smaller than the window size ({}x{}).".format(idx, px, py, window_size, window_size))
                cleanup(img)
                continue

            # Skip file if does not meet the aspect ratio requirement
            if py / px < pixel_ratio:
                if verbose:
                    print(f"[{idx}] [{scan_direction_full}] Skipping scan direction '{scan_direction_full}' for file '{file_name_without_ext}' due to low aspect ratio (py/px = {py/px:.2f}).")
                    #print(f"Skipping scan direction {scanDirection} for file {file_name_without_ext} due to low py/px ratio (ratio: {py/px:.2f}).\n")
                cleanup(img)
                continue
            
            # Here we rescale to a constant pixel density for ML training
            if resample:
                try:
                    # Calculate the original pixel density and rescale the image
                    pixel_density_orig = px / metadata['width']
                    img = resample_image_data(img, pixel_density, pixel_density_orig, pixel_limit=20000, verbose=verbose)
                except ZeroDivisionError:
                    if verbose:
                        print(f"[{idx}] [{scan_direction_full}] Skipping scan direction '{scanDirection}' for file '{file_name_without_ext}' due to zero width in metadata (division by zero error).")
                        #print(f"Skipping scan direction {scanDirection} for file {file_name_without_ext} due to zero width in metadata (division by zero error).")
                    continue  # Skip this image and go to the next
            
            # Get the dimensions of the resized image
            py, px = img.shape

            if verbose:
                print(f"[{idx}] [{scan_direction_full}] Image rescaled: ({px}x{py})")

            # Skip the file if the rescaled image has pixel numbers are fewer than the window size in either dimension
            if px < window_size or py < window_size:
                if verbose:
                    print(f"[{idx}] [{scan_direction_full}] Skipping rescaled image with size ({px}x{py}) as it is smaller than the window size ({window_size}x{window_size}).")
                    #print("{} Skipping image of size ({}x{}) because it is smaller than the window size ({}x{}).".format(idx, px, py, window_size, window_size))
                cleanup(img)
                continue    

            # Process image
            img = flatten_image_data(img, flatten_method)

            # Subtract image minimum and then scale.
            img = (img - np.min(img)) * data_scaling

            # Skip images with data outside the range [0,1]
            if np.max(img) > 1.0:
                if verbose:
                    print(f"[{idx}] [{scan_direction_full}] Skipping scan direction '{scan_direction_full}' for file '{file_name_without_ext}' due to data outside the range [0,1] (max = {np.max(img):.2f}).")
                    #print(f"Skipping scan direction {scanDirection} for file {file_name_without_ext} due to data outside range [0,1]: max= {np.max(img):.2f}).\n")
                continue
            else:
                if verbose:
                    #print('Original Image data range min: {}, image max: {}.'.format(np.min(img), np.max(img)))
                    print(f"[{idx}] [{scan_direction_full}] Original Image Data Range: Min = {np.min(img):.2f}, Max = {np.max(img):.2f}.")

            # NORMALISE ALL IMAGES to [0,1]
            try:
                # Try dividing by the max value in the image array
                img = img / np.max(img)
            except ZeroDivisionError:
                if verbose:
                    print(f"[{idx}] [{scan_direction_full}] Skipping scan direction '{scanDirection}' for file '{file_name_without_ext}' due to zero max value (division by zero error).")
                    #print(f"Skipping scan direction {scanDirection} for file {file_name_without_ext} due to zero max value (division by zero error).")
                continue  # Skip this image and go to the next
            
            if verbose:
                print(f"[{idx}] [{scan_direction_full}] Image Data Range Scaled: Min = {np.min(img):.2f}, Max = {np.max(img):.2f}.")

            # Get the subfolder of the MTRX data
            mtrx_path_no_filename = os.path.dirname(mtrx_path)

            mtrx_path_index = mtrx_path_no_filename.find('mtrx')
            relative_dir = mtrx_path_no_filename[mtrx_path_index + len('mtrx'):] if mtrx_path_index != -1 else ''

            if save_jpg: 
                # Create path for the JPG data save
                jpg_full_path = create_folder_path(jpg_path, sub_dir=relative_dir, collate=collate)

                # Save the whole image as JPG
                jpg_save_filename = file_name_without_ext + '_' + scan_direction + '.jpg'
                jpg_save_path = os.path.join(jpg_full_path, jpg_save_filename)
                if not verbose:
                    sys.stdout.write("\b")  # Moves back one character
                    sys.stdout.flush()
                    sys.stdout.write("j")  # Replace 
                    sys.stdout.flush()
                save_as_jpg(img, jpg_save_path, cmap=cmap, verbose=verbose)
                
                # Save the metadata as a text file
                jpg_txt_save_path = jpg_save_path.replace('.jpg', '.txt')
                if save_meta:
                    save_metadata(metadata, jpg_txt_save_path)

            if save_windows: #or return_windows:
                   
                # Create windows
                img_windows, coordinates = extract_image_windows(img, px=window_size, pitch=window_pitch)

                ## THIS WAS A TEST OF RENORMALISING THE WINDOWS. IT HAD TERRIBLE RESULTS.  
                # # Normalize the image windows: (image - min) / (max - min)
                # mins = np.min(img_windows, axis=(1, 2), keepdims=True)
                # maxs = np.max(img_windows, axis=(1, 2), keepdims=True)            
                # img_windows = (img_windows - mins) / (maxs - mins)

                # if return_windows:
                #     all_windows.append(np.array(img_windows))  # Append windows as numpy array
                #     all_metadata.append(metadata)

                if save_windows:
                    windows_base_save_file_name = file_name_without_ext + '_' + scan_direction
                
                    windows_relative_save_path = os.path.join(relative_dir, windows_base_save_file_name)   
                
                    # Create path for the WINDOWS data save
                    windows_full_path = create_folder_path(windows_path, sub_dir=windows_relative_save_path, collate=collate)
                    if save_window_jpgs:
                        windows_jpg_full_path = create_folder_path(windows_jpg_path, sub_dir=windows_relative_save_path, collate=collate)

                    # # Save windows
                    if together: 
                        if not verbose:
                            sys.stdout.write("\b")  # Moves back one character
                            sys.stdout.flush()
                            sys.stdout.write("t")  # Replace with a period
                            sys.stdout.flush()
                        save_image_windows_together(img_windows, coordinates, windows_full_path, base_filename=windows_base_save_file_name, verbose=verbose)
                    else:
                        if not verbose:
                            sys.stdout.write("\b")  # Moves back one character
                            sys.stdout.flush()
                            sys.stdout.write("i")  # Replace 
                            sys.stdout.flush()
                        save_image_windows_individually(img_windows, coordinates, windows_full_path, base_filename=windows_base_save_file_name, verbose=verbose)

                    if save_window_jpgs:
                        # Create path for the WINDOWS data save
                        windows_full_path_jpgs = create_folder_path(windows_jpg_path, sub_dir=windows_relative_save_path, collate=collate)
                        save_image_windows_as_jpg(img_windows, coordinates, windows_full_path_jpgs, base_filename=windows_base_save_file_name, verbose=verbose)

                    # Save the metadata as a text file 
                    save_path = os.path.join(windows_full_path, windows_base_save_file_name)
                    if save_meta:
                        save_metadata(metadata, save_path + '.txt')

                    # Clear variables to avoid memory problems
                    cleanup(img, img_windows)
                
                

    print()
    print("********************")
    print("Conversion complete.")
    print("********************\n")

# ============================================================================
# File IO
# ============================================================================

def list_files_by_extension(directory, extension=".Z_mtrx", verbose=False):
    """
    Recursively list all files with a specific extension in the given directory and its subdirectories.

    Parameters:
        directory (str): Path to the directory.
        extension (str): File extension to look for (e.g., ".Z_mtrx"). Default is ".Z_mtrx".
        verbose (bool): If True, prints the list of files and the total count.

    Returns:
        file_paths_with_ext (list): List of file paths with the specified extension.
        num_files (int): The number of files found with the specified extension.
    """

    # Initialize an empty list to store file paths with the specified extension
    file_paths_with_ext = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has the specified extension
            if file.endswith(extension):
                file_paths_with_ext.append(os.path.join(root, file))
    
    # Sort the list to ensure consistent ordering
    file_paths_with_ext.sort()

    # Get the number of files
    num_files = len(file_paths_with_ext)

    if verbose:
        for index, filepath in enumerate(file_paths_with_ext, start=1):
            print(f"{index}. {filepath}")
        print()
    
    print("Found {} files with extension '{}' in directory:".format(num_files,extension))
    print(directory)
    
    file_paths_with_ext.sort()

    # Return the list of file paths and the total number of files
    return file_paths_with_ext, num_files


def load_mtrx_data(mtrx_path):
    """
    Load MTRX image data from a given file path.

    Parameters:


    Returns:
        tuple: The loaded image data and the associated message file.
    """
    
    if not os.path.exists(mtrx_path):
        print(f"File not found: {mtrx_path}")
        return None, None

    try:
        # Suppress print while loading the data
        imgAll, message_file = suppress_print(mtrx.open, mtrx_path)
        #print(message_file)  # To print the message 
        return imgAll, message_file
    except Exception as e:
        print(f"Error in opening {mtrx_path}: {e}")
        return None, None


def create_folder_path(base_path, sub_dir=None, collate=False):
    """
    Create folder path for saving data.

    Parameters:
        base_path (str): The base directory path where the data should be saved.
        sub_dir (str, optional): A sub-directory structure to be appended to the base path.
                                 Ignored if `collate` is True.
        collate (bool): If True, data will be saved in the base path.
                        If False, data will be saved in the sub-directory under the base path.

    Returns:
        str: The full path of the created directory.
    """
    if sub_dir:
        # Strip leading '/' if it exists in sub_dir
        sub_dir = sub_dir.lstrip('/')

    if collate:
        full_path = base_path
    else:
        full_path = os.path.join(base_path, sub_dir)

    os.makedirs(full_path, exist_ok=True)

    return full_path


def delete_data_folders(data_path, subdirectories=None, override=False):
    """
    Delete specified subdirectories within the given data path.

    Parameters:
        data_path (str): The base path where the folders are located.
        subdirectories (str or list, optional): Subdirectory or list of subdirectories to delete. 
                                                If None, defaults to ["jpg", "windows", "raw"].
        override (bool, optional): If True, bypasses user confirmation and performs the deletion directly.
    """
    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Handle subdirectories being either a string or list
    if isinstance(subdirectories, str):
        folders_to_delete = [subdirectories]
    elif isinstance(subdirectories, list):
        folders_to_delete = subdirectories
    else:
        print("Error: please specify subdirectories to be deleted by adding 'subdirectories=...'")
        return

    # Create full paths for each folder to delete
    full_paths = [os.path.join(data_path, folder) for folder in folders_to_delete]

    # Check which folders exist
    existing_folders = [path for path in full_paths if os.path.exists(path)]
    
    if not existing_folders:
        print("No target folders found to delete.")
        return

    # Check that each folder to be deleted is a subdirectory of the user's home directory
    for folder in existing_folders:
        abs_folder = os.path.abspath(folder)

        # Ensure the folder is under the user's home directory
        if not abs_folder.startswith(home_directory):
            print(f"Error: The folder {folder} is not a subdirectory of the user's home directory. Aborting deletion.")
            return
        
        # Ensure we are not deleting directly under the home directory
        if abs_folder == home_directory or os.path.dirname(abs_folder) == home_directory:
            print(f"Error: The folder {folder} is at the top level of the user's home directory. Aborting deletion.")
            return

    if not override:
        # message
        print("The following folders and all their contents will be permanently deleted and cannot be undone:")
        for folder in existing_folders:
            print(f"- {folder}")

        # Confirmation prompt
        confirmation = input("Type 'yes' to confirm deletion of these folders: ").strip()
        
        if confirmation != "yes":
            print("Deletion cancelled.")
            return

    # Delete the folders
    for folder in existing_folders:
        for root, dirs, files in os.walk(folder, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(folder)
        print(f"Deleted: {folder}")

    print("All specified folders have been successfully deleted.")


def create_new_data_path(base_path, job_name, include_date=False):
    """
    Create a new data path by appending a unique subdirectory to the base path.
    
    Parameters:
        base_path (str): The base directory for storing data.
        job_name (str): The name for the subdirectory (e.g., job identifier).

    Returns:
        str: The path of the newly created subdirectory.
    """
    # Create a unique subdirectory using timestamp or job name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if include_date:
        new_data_path = os.path.join(base_path, f"{job_name}_{timestamp}")
    else:
        new_data_path = os.path.join(base_path, f"{job_name}")

    # Create the directory
    os.makedirs(new_data_path, exist_ok=True)

    return new_data_path


def load_coordinates_file(windows_predict_path):
    """
    Loads a coordinates text file from the given path and returns it as a NumPy array.

    Parameters:
    - windows_predict_path: str, the path to the directory containing the target file.

    Returns:
    - numpy array containing the coordinates data (excluding the header).
    """
    # Extract the last part of the given path
    base_name = os.path.basename(windows_predict_path)
    
    if 'coordinates' not in base_name:
        # Construct the filename by appending '_coordinates.txt' to the base name
        coordinates_file_name = f"{base_name}_coordinates.txt"
    else:
        coordinates_file_name = windows_predict_path

    # Construct the full path to the coordinates file
    coordinates_file_path = os.path.join(windows_predict_path, coordinates_file_name)

    # Check if the file exists
    if not os.path.exists(coordinates_file_path):
        raise FileNotFoundError(f"The file {coordinates_file_path} does not exist.")

    # Load the data from the file, skipping the header line, and ensuring data types are integers
    try:
        coordinates_data = np.loadtxt(coordinates_file_path, skiprows=1, dtype=int)
    except ValueError as e:
        raise ValueError(f"Error loading the coordinates file. Ensure the file format is correct: {e}")

    return coordinates_data[:, [1, 2]]


def find_bottom_layer_folders(path):
    """
    Recursively searches through a given directory path and returns a list of all 
    bottom-layer (leaf) folders. A bottom-layer folder is defined as a folder that 
    contains no further subdirectories.
    
    Parameters:
    path (str): The base directory path to start the recursive search.
    
    Returns:
    List[str]: A list of absolute paths to the bottom-layer folders.
    """
    bottom_layer_folders = []

    for root, dirs, files in os.walk(path):
        # If 'dirs' is empty, it means 'root' is a bottom layer folder
        if not dirs:
            bottom_layer_folders.append(root)

    bottom_layer_folders.sort()
    
    return bottom_layer_folders


# ============================================================================
# 
# ============================================================================


def suppress_print(func, *args, **kwargs):
    """
    Suppresses print statements during the execution of a function.
    
    Parameters:
        func: The function to execute.
        *args, **kwargs: Arguments and keyword arguments for the function.

    Returns:
        The result of the function call.
    """
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null
    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout  # Restore original stdout


def extract_regulation_parameters_from_mtrx_data(mtrx_data):
    """
    Extract specific regulation parameters from an already loaded MtrxData object.

    Parameters:
        mtrx_data (MtrxData): An instance of the MtrxData class with data loaded.

    Returns:
        dict: A dictionary with specific parameters renamed and organized.
    """
    # Attempt to retrieve experiment parameters from the mtrx_data object
    try:
        experiment_params = mtrx_data.get_experiment_element_parameters()
    except AttributeError:
        print("The 'MtrxData' object does not have a method 'get_experiment_element_parameters'.")
        return {}

    # The first element of the tuple contains the actual parameter list
    if isinstance(experiment_params, tuple) and len(experiment_params) > 0:
        param_list = experiment_params[0]

        # Create a dictionary to store the extracted parameters
        extracted_params = {}

        # Define the mapping of original keys to new keys
        param_mapping = {
            'GapVoltageControl.Voltage': 'bias_forward',
            'GapVoltageControl.Alternate_Voltage': 'bias_backward',
            'Regulator.Setpoint_1': 'current_forward',
            'Regulator.Alternate_Setpoint_1': 'current_backward',
            'GapVoltageControl.Enable_Alternate_Voltage': 'enable_alternate_voltage'
        }

        # Extract the specific parameters
        for item in param_list:
            if len(item) == 3:
                param_name, value, unit = item
                if param_name in param_mapping:
                    new_key = param_mapping[param_name]
                    extracted_params[new_key] = value
                    if param_name != 'GapVoltageControl.Enable_Alternate_Voltage':
                        extracted_params[f"{new_key}_unit"] = unit

        # Check the 'enable_alternate_voltage' flag
        if not extracted_params.get('enable_alternate_voltage', False):
            # If the alternate voltage is disabled, set the alternate values to the forward values
            extracted_params['bias_backward'] = extracted_params.get('bias_forward')
            extracted_params['bias_backward_unit'] = extracted_params.get('bias_forward_unit')
            extracted_params['current_backward'] = extracted_params.get('current_forward')
            extracted_params['current_backward_unit'] = extracted_params.get('current_forward_unit')

        # Remove the 'enable_alternate_voltage' entry from the final output
        extracted_params.pop('enable_alternate_voltage', None)

        return extracted_params

    else:
        print("The experiment parameters are in an unrecognized format.")
        return {}


def save_as_jpg(img, save_path, cmap='gray', verbose=False):
    """
    Save the image data as a JPG file with an optional colormap.

    Parameters:
        img (np.array): 2D numpy array of image data.
        save_path (str): Full path where the JPG file will be saved.
        cmap (str): Matplotlib colormap name (default is 'gray').
        verbose (bool): If True, print out additional information.
    """
    if img.dtype != np.uint8:
        # Scale the float32 image data (assuming range [0, 1]) to uint8 range [0, 255]
        img = (img * 255).astype(np.uint8)

    # Apply the colormap using matplotlib.colormaps
    colormap = plt.colormaps[cmap]  # Get the colormap by name
    colored_img = colormap(img / 255.0)  # Normalize to [0, 1] for colormap
    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB and scale to [0, 255]

    # Create an Image object from the numpy array and save as JPEG
    image = Image.fromarray(colored_img)
    image.save(save_path, format='JPEG')

    if verbose:
        print('Saved full image as jpg: {}'.format(os.path.basename(save_path)))

        
def save_metadata(metadata, save_path):
    """
    Save the image metadata (including width, height, angle, scan direction, bias, and current) to a text file.

    Parameters:
        metadata (dict): Dictionary containing metadata keys such as 'width', 'height', 'angle',
                         'scan_direction', 'bias', 'bias_unit', 'current', 'current_unit'.
        save_path (str): Full path where the text file will be saved, with '.txt' extension.
    """
    with open(save_path, 'w') as file:
        # Define the width for the columns
        col_width_key = 15
        col_width_value = 20
        col_width_unit = 12

        # Save scan direction if present
        if 'scan_direction' in metadata:
            file.write(f"{'scan_direction'.ljust(col_width_key)}{metadata['scan_direction'].rjust(col_width_value)}\n")

        # Write each line with formatted spacing and units
        file.write(f"{'width'.ljust(col_width_key)}{str(metadata.get('width', '')).rjust(col_width_value)}{'nm'.rjust(col_width_unit)}\n")
        file.write(f"{'height'.ljust(col_width_key)}{str(metadata.get('height', '')).rjust(col_width_value)}{'nm'.rjust(col_width_unit)}\n")
        file.write(f"{'angle'.ljust(col_width_key)}{str(metadata.get('angle', '')).rjust(col_width_value)}{'degrees'.rjust(col_width_unit)}\n")

        # Optionally save bias and current if they are present
        if 'bias' in metadata and 'bias_unit' in metadata:
            file.write(f"{'bias'.ljust(col_width_key)}{str(metadata['bias']).rjust(col_width_value)}{metadata['bias_unit'].rjust(col_width_unit)}\n")

        if 'current' in metadata and 'current_unit' in metadata:
            file.write(f"{'current'.ljust(col_width_key)}{str(metadata['current']).rjust(col_width_value)}{metadata['current_unit'].rjust(col_width_unit)}\n")


def extract_image_windows(image, px=128, pitch=128):

    # Get image dimensions
    height, width = image.shape
    #print('EXTRRACTING WINDOWS. ORIG IMAGE: Height {} Width {}'.format(height, width))

    # Calculate the number of windows in each dimension
    num_windows_y = max(1, (height - px) // pitch + 1)
    num_windows_x = max(1, (width - px) // pitch + 1)

    # Generate list of window start positions
    y_positions = [y * pitch for y in range(num_windows_y)]
    x_positions = [x * pitch for x in range(num_windows_x)]

    # Adjust start positions to ensure full windows at the edges
    if y_positions[-1] + px > height:
        y_positions[-1] = height - px
    if x_positions[-1] + px > width:
        x_positions[-1] = width - px

    # Extract windows and record coordinates
    windows = []
    coordinates = []
    window_number = 0
    for y in y_positions:
        for x in x_positions:
            # Extract windows with consistent size
            window = image[y:y+px, x:x+px]
            windows.append(window)
            coordinates.append([window_number, x, y])
            window_number += 1

            # if window.shape[0] != 32 or window.shape[1] != 32:
            #     print('\n\n*************\n window number {}: {} {}\n\n'.format(window_number, window.shape[0],window.shape[1]))
                
            #     plt.imshow(window)
            #     plt.show()

    return np.array(windows), np.array(coordinates)


def save_image_windows_individually(img_windows, coordinates, save_dir, base_filename='window', verbose=False):
    """
    Save each image window from a 3D numpy array to separate raw binary files (NumPy format)
    using parallel processing. Also save the coordinates of each window to a text file.

    Parameters:
        img_windows (np.array): 3D numpy array of image windows (shape: [num_windows, height, width]).
        coordinates (np.array): 2D numpy array of window coordinates (shape: [num_windows, 3]).
        save_dir (str): Directory where the raw data will be saved.
        base_filename (str): Base name for each window file.
        verbose (bool): If True, print out additional information.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Helper function to save an individual window
    def save_window(i):
        filename = f"{base_filename}_{i:05d}.npy"
        save_path = os.path.join(save_dir, filename)
        window = img_windows[i].astype(np.float32) if img_windows[i].dtype != np.float32 else img_windows[i]
        np.save(save_path, window)

    # Save all image windows in parallel
    num_windows = img_windows.shape[0]
    with ThreadPoolExecutor() as executor:
        list(executor.map(save_window, range(num_windows)))

    # Save the coordinates as a text file with formatted columns
    coordinates_filename = f"{base_filename}_coordinates.txt"
    coordinates_path = os.path.join(save_dir, coordinates_filename)
    fmt = '%05d          %-15s%-15s'
    header = f"{'Window_Number':<15}{'X_Position':<15}{'Y_Position':<15}"
    np.savetxt(coordinates_path, coordinates, fmt=fmt, header=header, comments='')

    if verbose:
        print(f'Saved {num_windows} windows\n')
        print(f'Saved coordinates to {coordinates_path}\n')


def save_image_windows_as_jpg(img_windows, coordinates, save_dir, base_filename='window', verbose=False):
    """
    Save each image window from a 3D numpy array to separate JPG files
    using parallel processing. Also save the coordinates of each window to a text file.

    Parameters:
        img_windows (np.array): 3D numpy array of image windows (shape: [num_windows, height, width]).
        coordinates (np.array): 2D numpy array of window coordinates (shape: [num_windows, 3]).
        save_dir (str): Directory where the JPG files will be saved.
        base_filename (str): Base name for each window file.
        verbose (bool): If True, print out additional information.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Helper function to save an individual window
    def save_window(i):
        filename = f"{base_filename}_{i:05d}.jpg"
        save_path = os.path.join(save_dir, filename)
        window = img_windows[i]
        # Normalize the window to [0, 255] and convert to uint8
        window = ((window - window.min()) / (window.max() - window.min()) * 255).astype(np.uint8)
        # Save the window as a JPG image
        Image.fromarray(window).save(save_path)

    # Save all image windows in parallel
    num_windows = img_windows.shape[0]
    with ThreadPoolExecutor() as executor:
        list(executor.map(save_window, range(num_windows)))

    # Save coordinates to a text file
    coordinates_path = os.path.join(save_dir, "coordinates.txt")
    np.savetxt(coordinates_path, coordinates, fmt="%d")

    if verbose:
        print(f"Saved {num_windows} windows as JPG files\n")
        print(f"Saved coordinates to {coordinates_path}\n")


def save_image_windows_together(img_windows, coordinates, save_dir, base_filename='window', verbose=False):
    """
    Save each image window from a 3D numpy array to separate raw binary files (NumPy format).
    Also save the coordinates of each window to a text file.

    Parameters:
        img_windows (np.array): 3D numpy array of image windows (shape: [num_windows, height, width]).
        coordinates (np.array): 2D numpy array of window coordinates (shape: [num_windows, 3]).
        save_dir (str): Directory where the raw data will be saved.
        base_filename (str): Base name for each window file.
        verbose (bool): If True, print out additional information.
        batch_save (bool): If True, save all windows in one batch file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save all image windows in a single .npy file
    save_path = os.path.join(save_dir, f"{base_filename}_all_windows.npy")
    np.save(save_path, img_windows)
    if verbose:
        num_windows = len(img_windows)
        print(f'Saved {num_windows} windows to {save_path}')

    # Save the coordinates as a text file with formatted columns
    coordinates_filename = f"{base_filename}_coordinates.txt"
    coordinates_path = os.path.join(save_dir, coordinates_filename)
    fmt = '%05d          %-15s%-15s'
    header = f"{'Window_Number':<15}{'X_Position':<15}{'Y_Position':<15}"
    np.savetxt(coordinates_path, coordinates, fmt=fmt, header=header, comments='')
    if verbose:
        print(f'Saved coordinates to {coordinates_path}\n')




def load_n_numpy_files(file_paths, N, mode='sequential', start_index=0):
    """
    Load N 2D numpy arrays from disk and return as a stacked array.

    Parameters:
        file_paths (list of str): List of full paths to .npy files.
        N (int): Number of files to load.
        mode (str): Mode of loading files. Options are 'sequential' or 'random'.
        start_index (int): Starting index for loading files when mode is 'sequential'.

    Returns:
        np.array: Loaded numpy arrays stacked together with shape [N, px, py].
    """
    # Ensure N does not exceed the length of file_paths
    N = min(N, len(file_paths))

    if mode == 'random':
        # Select N random file paths
        selected_paths = random.sample(file_paths, N)
    elif mode == 'sequential':
        # Ensure start_index is within bounds
        start_index = min(start_index, len(file_paths) - 1)
        selected_paths = file_paths[start_index:start_index + N]
    else:
        raise ValueError("Invalid mode. Use 'sequential' or 'random'.")

    # Load the selected numpy arrays from the provided file paths
    loaded_arrays = []
    for file_path in selected_paths:
        array = np.load(file_path)        
        loaded_arrays.append(array)

    # Stack the loaded arrays into a single numpy array with shape [N, px, py]
    stacked_array = np.stack(loaded_arrays, axis=0)

    return stacked_array


# ============================================================================
# Image manipulation and viewing
# ============================================================================

def flatten_image_with_row_and_plane(image):
    """
    Flatten an image by performing line-wise mean subtraction 
    followed by a global first-order plane subtraction.

    Parameters:
        image (numpy.ndarray): Input 2D image array.

    Returns:
        numpy.ndarray: Flattened image.
    """
    # Step 1: Line-wise mean subtraction
    row_means = np.mean(image, axis=1, keepdims=True)
    image_line_flattened = image - row_means

    # Step 2: Fit a first-order plane to the entire image
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x_flat, y_flat = x.ravel(), y.ravel()
    z_flat = image_line_flattened.ravel()

    # Solve for plane coefficients (z = ax + by + c)
    A = np.vstack([x_flat, y_flat, np.ones_like(x_flat)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    a, b, c = coeffs

    # Create the plane to subtract
    plane = (a * x + b * y + c)

    # Subtract the plane
    image_plane_flattened = image_line_flattened - plane

    return image_plane_flattened


def flatten_image_by_row_advanced(image, polyorder=0, window_length=11):
    """
    Not working currently
    Flatten an image row-wise using a Savitzky-Golay filter.

    Parameters:
        image (numpy.ndarray): Input 2D image array.
        polyorder (int): Polynomial order for the filter.
        window_length (int): Length of the filter window (must be odd).

    Returns:
        numpy.ndarray: Row-flattened image.
    """
    flattened_image = np.empty_like(image)
    for i, row in enumerate(image):
        baseline = savgol_filter(row, window_length=window_length, polyorder=polyorder)
        flattened_image[i] = row - baseline
    return flattened_image


def flatten_by_row_mean_and_slope(image):
    """
    Flatten an image by subtracting the row mean and slope from each row.

    Parameters:
        image (numpy.ndarray): Input 2D image array.

    Returns:
        numpy.ndarray: Row-flattened image.
    """
    flattened_image = np.empty_like(image)
    x = np.arange(image.shape[1])  # x-coordinates for fitting

    for i, row in enumerate(image):
        # Perform a linear fit (y = mx + c)
        slope, intercept = np.polyfit(x, row, deg=1)
        # Calculate the linear trend
        trend = slope * x + intercept
        # Subtract the trend from the row
        flattened_image[i] = row - trend

    return flattened_image


def flatten_by_row_mean(image):
    """
    Flatten an image by subtracting the row mean from each row.

    Parameters:
        image (numpy.ndarray): Input 2D image array.

    Returns:
        numpy.ndarray: Row-flattened image.
    """
    row_means = np.mean(image, axis=1, keepdims=True)
    flattened_image = image - row_means
    return flattened_image


def flatten_image_data(img, flatten_method='iterate_mask'):
    """
    Flatten image data using the specified method.

    Parameters:
        img (np.array): 2D numpy array of image data.
        flatten_method (str): Method to flatten the image.
                              Options: 'none', 'iterate_mask', 'poly_xy'.

    Returns:
        np.array: Flattened image data.
    """
    if flatten_method == "iterate_mask":
        img, mask, n = spiepy.flatten_by_iterate_mask(img)
    elif flatten_method == "poly_xy":
        img, _ = spiepy.flatten_poly_xy(img, deg=1)
    elif flatten_method == "row_mean":
        img = flatten_by_row_mean(img)
    elif flatten_method == 'row_mean_and_slope':
        img = flatten_by_row_mean_and_slope(img)

    # Remove any remaining constant offset
    img = img - np.min(img)
    return img


def resample_image_data(img, pixel_density, pixel_density_orig, pixel_limit=5000, verbose=True):
    """
    Resample image data to match the specified pixel density.

    Parameters:
        img (np.array): 2D numpy array of image data.
        real_width (int): Real width of the image in nanometers.
        pixel_density (float): Desired pixel density (px/nm).
        pixel_density_orig (float): Original pixel density of the image (px/nm).
        verbose (bool): If True, print out additional information.

    Returns:
        np.array: Resampled image data.
    """
    px, py = img.shape[1], img.shape[0]

    if pixel_density_orig != pixel_density:
        px = round(px * pixel_density / pixel_density_orig)
        py = round(py * pixel_density / pixel_density_orig)

        if px > pixel_limit or py > pixel_limit:
            if verbose: 
                print(f"\n*** The rescaled image would have dimension {px,py}. This is greature than the specified pixel limit {pixel_limit,pixel_limit}. Skipping image")
            img = np.zeros((2,2),dtype=float)
        else:
            img = cv2.resize(img, (px, py), interpolation=cv2.INTER_AREA)

    return img


def display_window_selection(windows, cols=3, total_width=15, title=None):
    """
    Displays a selection of image windows in a grid. The number of columns is specified by 'cols',
    and the total number of windows is cols^2. The total image width is fixed, and the height is adjusted
    dynamically based on the number of rows. Optionally, you can provide a plot title.

    Parameters:
        windows (numpy array): An array of image windows to be displayed.
        cols (int): The number of columns. The total number of windows will be cols^2.
        total_width (float): The total width of the grid in inches.
        title (str): Optional plot title. Default is None (no title).
    """
    
    # Check if data is in CNN format (N, px, py, grayscale/RGB)
    if len(windows.shape) == 4:
        # If grayscale (1 channel) or RGB (3 channels), ignore the last channel
        if windows.shape[-1] == 1 or windows.shape[-1] == 3:
            windows = windows[..., 0]  # Take the first channel (ignoring RGB or grayscale channel)

    # Calculate the total number of windows (cols^2)
    num_windows = cols ** 2

    # Calculate the number of rows
    num_rows = cols

    # Calculate the height dynamically to keep the total width fixed
    aspect_ratio = windows.shape[1] / windows.shape[2] if len(windows.shape) >= 3 else 1  # height/width ratio
    total_height = total_width * (num_rows / cols) * aspect_ratio

    # Plot each window
    fig, axes = plt.subplots(num_rows, cols, figsize=(total_width, total_height))
    axes = axes.ravel()

    for i in range(min(num_windows, windows.shape[0])):  # Ensure we don't exceed available windows
        axes[i].imshow(windows[i], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(min(num_windows, len(axes)), len(axes)):
        axes[i].axis('off')

    space = 0.05
    # Adjust spacing between windows
    plt.subplots_adjust(
        hspace=space, 
        wspace=space,
        left=space, 
        bottom=space,
        right=1 - space, 
        top=1 - space
    )

    # Add plot title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    plt.show()


def reconstruct_image(windows, windows_coordinates, window_size):
    """
    Reconstruct the original image from overlapping windows.

    Args:
        windows (numpy.ndarray): Array of windowed images of shape (N, px, py).
        windows_coordinates (numpy.ndarray): Array of coordinates for each window of shape (N, 2),
                                             where the columns represent [x, y].
        window_size (tuple): Size of each window (px, py).

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    px, py = (window_size, window_size)

    # Determine the dimensions of the final reconstructed image
    max_x = max(windows_coordinates[:, 0]) + px
    max_y = max(windows_coordinates[:, 1]) + py

    # Create an array to store the reconstructed image
    reconstructed_image = np.zeros((max_y, max_x))

    # Iterate over all windows and place them in their respective positions
    for i in range(len(windows)):
        # Extract x and y coordinates (ignoring window_number)
        x, y = windows_coordinates[i, 0], windows_coordinates[i, 1]
        window = windows[i]

        # Replace the data in the appropriate location in the reconstructed image
        reconstructed_image[y:y + py, x:x + px] = window

    return reconstructed_image



def reconstruct_cluster_image(windows_coordinates, window_size, cluster_labels):
    """
    Reconstruct the original image from overlapping windows using cluster labels.

    Args:
        windows_coordinates (numpy.ndarray): Array of coordinates for each window of shape (N, 2),
                                             where the columns represent [x, y].
        window_size (tuple): Size of each window (px, py).
        cluster_labels (numpy.ndarray): Array of predicted cluster labels for each window of shape (N,).

    Returns:
        numpy.ndarray: Reconstructed image with cluster labels.
    """
    px, py = (window_size, window_size)

    px = windows_coordinates[1, 0] - windows_coordinates[0, 0]
    py = px

    # Determine the dimensions of the final reconstructed image
    max_x = max(windows_coordinates[:, 0]) + px
    max_y = max(windows_coordinates[:, 1]) + py

    # Create an array to store the reconstructed image based on cluster labels
    reconstructed_image = np.zeros((max_y, max_x))
    weight_matrix = np.zeros((max_y, max_x))

    # Iterate over all windows and place cluster labels in their respective positions
    for i in range(len(cluster_labels)):
        # Extract x and y coordinates
        x, y = windows_coordinates[i, 0], windows_coordinates[i, 1]
        cluster_value = cluster_labels[i]

        # Set all values in the window to the cluster label
        reconstructed_image[y:y + py, x:x + px] += cluster_value
        weight_matrix[y:y + py, x:x + px] += 1

    # Avoid division by zero by setting non-overlapping regions to 1
    weight_matrix[weight_matrix == 0] = 1

    # Average overlapping regions by dividing by the weight matrix
    reconstructed_image /= weight_matrix

    return reconstructed_image



# ============================================================================
#Adam
# ============================================================================

def save_feature_windows_together(img_windows, coordinates, save_dir, base_filename='feature_window', verbose=False):
    """
    Save each image window from a 3D numpy array to separate raw binary files (NumPy format).
    Also save the coordinates of each window to a text file.

    Parameters:
        img_windows (np.array): 3D numpy array of image windows (shape: [num_windows, height, width]).
        coordinates (np.array): 2D numpy array of window coordinates (shape: [num_windows, 3]).
        save_dir (str): Directory where the raw data will be saved.
        base_filename (str): Base name for each window file.
        verbose (bool): If True, print out additional information.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all image windows in a single .npy file
    save_path = os.path.join(save_dir, f"{base_filename}_all_windows.npy")
    np.save(save_path, img_windows)
    if verbose:
        num_windows = len(img_windows)
        print(f'Saved {num_windows} windows to {save_path}')

    # Save the coordinates as a text file with formatted columns
    coordinates_filename = f"{base_filename}_coordinates.txt"
    coordinates_path = os.path.join(save_dir, coordinates_filename)
    #fmt = '%05d          %-15s%-15s'
    header = f"{'Y_Position':<15}{'X_Position':<15}"
    np.savetxt(coordinates_path, coordinates, header=header, comments='')
    if verbose:
        print(f'Saved coordinates to {coordinates_path}\n')


def save_feature_windows_coords_and_labels(windows, coords, labels, save_dir, base_filename="feature_window", verbose=False):
    """
    Save each image feature window from a 3D numpy array to separate raw binary files (NumPy format).
    Also save the coordinates and label of each feature window to a text file. The coordinates file will have
    three columns: Y_Position, X_Position, and Label_Number.

    Parameters:
        windows (np.array): 3D numpy array of image windows (shape: [num_windows, height, width]).
        coords (np.array): 2D numpy array of window coordinates (shape: [num_windows, 3]).
        save_dir (str): Directory where the raw data will be saved.
        base_filename (str): Base name for each window file.
        verbose (bool): If True, print out additional information.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all image windows in a single .npy file
    save_path = os.path.join(save_dir, f"{base_filename}.npy")
    np.save(save_path, windows)
    if verbose:
        num_windows = len(windows)
        print(f'Saved {num_windows} windows to {save_path}')

    # Save the coordinates as a text file with formatted columns
    coordinates_filename = f"{base_filename}_coordinates.txt"
    coordinates_path = os.path.join(save_dir, coordinates_filename)
    #fmt = '%05d          %-15s%-15s'
    
    # Ensure sorted_labels is (210, 1)
    labels_col = np.expand_dims(labels, axis=1) if labels.ndim == 1 else labels

    # Combine
    coords_labels = np.hstack((coords, labels_col))  # shape (N, 3)
    
    header = f"{'Y_Position':<15}{'X_Position':<15}{'Label_Number'}"
    np.savetxt(coordinates_path, coords_labels, header=header, comments='')
    if verbose:
        print(f'Saved coordinates to {coordinates_path}\n')