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

import matplotlib.pyplot as plt
from matplotlib import colormaps  # Import the new colormap module



# ============================================================================
# General Image stuff
# ============================================================================

def load_jpg_as_norm_numpy(filepath):
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



# ============================================================================
# Colormap stuff
# ============================================================================

def display_colormap_and_rgb(cmap, n_colors=256, image=None):
    """
    Displays a colormap as a vertical bar, its RGB components, and optionally an image.
    
    Parameters:
        cmap: A preloaded colormap object (e.g., matplotlib.colors.Colormap).
        n_colors (int): Number of discrete colors to sample from the colormap (default: 256).
        image (array-like, optional): Image to display in the third panel (default: None).
    """
    # Generate the data for the colormap
    gradient = np.linspace(0, 1, n_colors).reshape(-1, 1)  # Gradient for the colormap
    colors = cmap(np.linspace(0, 1, n_colors))  # Sample the RGB values

    # Determine the layout based on whether an image is provided
    n_panels = 3 if image is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 3] + ([3] if image is not None else [])})

    # Plot the colormap as a vertical bar
    axes[0].imshow(gradient, aspect='auto', cmap=cmap, origin='lower')  # Set origin to 'lower'
    axes[0].axis('off')  # Hide axes for the colormap
    axes[0].set_title(f"Colormap", fontsize=12)

    # Plot the RGB curves
    x = np.linspace(0, 1, n_colors)
    axes[1].plot(x, colors[:, 0], color='red', label='Red')
    axes[1].plot(x, colors[:, 1], color='green', label='Green')
    axes[1].plot(x, colors[:, 2], color='blue', label='Blue')
    axes[1].set_xlabel('Normalized Position', fontsize=10)
    axes[1].set_ylabel('Intensity', fontsize=10)
    axes[1].set_title(f"RGB Components", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Optionally plot the image
    if image is not None:
        axes[2].imshow(image, cmap=cmap if image.ndim == 2 else None)
        axes[2].axis('off')  # Hide axes for the image
        axes[2].set_title("Image", fontsize=12)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_colormaps_image_grid(image, ncols=None, figwidth=20, title_fontsize=12):
    """
    Plots the same image using all available Matplotlib colormaps in a grid.
    
    Parameters:
        image (array-like): The image data (2D array) to display.
        ncols (int, optional): Number of columns in the grid. Default is the square root of the number of colormaps.
        figwidth (float): Fixed width of the figure. Default is 20.
        title_fontsize (int): Font size for image titles. Default is 12.
    """
    colormaps = plt.colormaps()  # Get all available colormaps
    n_images = len(colormaps)
    
    # Calculate grid dimensions
    if ncols is None:
        ncols = round(np.sqrt(n_images))
    nrows = (n_images + ncols - 1) // ncols  # Rows needed for the grid

    # Determine individual panel size and figure height
    panel_width = figwidth / ncols  # Width of each panel
    panel_height = panel_width  # Ensure square panels
    figheight = panel_height * nrows  # Total figure height
    
    # Create the figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))
    axes = axes.flatten()  # Flatten the grid for easy indexing

    for i, ax in enumerate(axes):
        if i < n_images:
            cmap = colormaps[i]
            ax.imshow(image, cmap=cmap)
            ax.set_title(cmap, fontsize=title_fontsize)  # Set font size for titles
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Hide any unused subplots

    plt.tight_layout()
    plt.show()