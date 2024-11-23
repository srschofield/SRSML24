#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for scanning probe images.
    
@author: Steven R. Schofield 

Created November 2024

"""

# ============================================================================
# Module dependencies
# ============================================================================

from PIL import Image
import numpy as np


# ============================================================================
# General Image stuff
# ============================================================================

def load_grayscale_image(filepath):
    """
    Load a JPEG image, convert it to grayscale, and return a flat 2D NumPy array.
    
    Parameters:
        filepath (str): Path to the JPEG image file.
        
    Returns:
        np.ndarray: A 2D array (grayscale) of type float32 in the range [0, 1].
    """
    # Load the image using PIL
    image = Image.open(filepath)
    
    # Convert the image to grayscale
    grayscale_image = image.convert("L")  # "L" mode = 8-bit grayscale
    
    # Convert to a NumPy array
    grayscale_array = np.asarray(grayscale_image, dtype=np.float32)
    
    # Normalize the array to the range [0, 1]
    normalized_array = grayscale_array / 255.0
    
    return normalized_array
