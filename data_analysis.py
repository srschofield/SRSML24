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

from curses import meta
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

from PIL import Image, ImageDraw, ImageFont

import re

import matplotlib.font_manager as fm



def radial_profile(img, center, cmap='gist_heat'):
    """
    Compute and display a radially averaged profile from a given centre point.

    Parameters
    ----------
    img : 2D np.array
    center : tuple (x, y) in pixel coordinates
    cmap : str
    
    Returns
    -------
    radii : np.array    radial distances in pixels
    profile : np.array  mean intensity at each radius
    """
    cx, cy = center

    ny, nx = img.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)

    max_radius = min(cx, cy, nx - cx, ny - cy)
    radii = np.arange(0, max_radius)
    profile = np.array([img[r == rad].mean() for rad in radii])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img, cmap=cmap, origin='lower')
    axes[0].plot(cx, cy, 'w+', markersize=12, markeredgewidth=2)
    axes[0].add_patch(plt.Circle((cx, cy), max_radius, color='white', fill=False, 
                                  linewidth=1, linestyle='--'))
    axes[0].set_title(f'Centre: ({cx}, {cy})')
    axes[0].axis('off')

    axes[1].plot(radii, profile)
    axes[1].set_xlabel('Radius (px)')
    axes[1].set_ylabel('Mean intensity')
    axes[1].set_title('Radial profile')
    plt.tight_layout()
    plt.show()

    return radii, profile