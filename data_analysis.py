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


from concurrent.futures import ThreadPoolExecutor

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from PIL import Image, ImageDraw, ImageFont

import re

import matplotlib.font_manager as fm



def pixel_size_from_metadata(img, metadata, direction='FU'):
    """
    Compute the physical pixel size (nm/pixel) from MTRX scan metadata.

    Parameters
    ----------
    img       : 2D np.array  — the loaded image array
    metadata  : dict         — returned by dp.load_image_for_processing()
    direction : str          — scan direction key, e.g. 'FU', 'BU', 'FD', 'BD'

    Returns
    -------
    pixel_size : float  — nm per pixel (x-axis; assumes square pixels)
    """
    m = metadata[direction]
    nx = img.shape[1]
    width_nm = m['width']
    if width_nm == 0:
        raise ValueError(
            f"metadata['{direction}']['width'] is 0 — scan width could not be "
            "read from the MTRX file.")
    pixel_size = width_nm / nx
    print(f"pixel_size_from_metadata: {width_nm} nm / {nx} px = {pixel_size:.4f} nm/px")
    return pixel_size


def pick_and_profile(img, cmap='gist_heat', pixel_size=None):
    """
    Display an image and compute a radial profile from a clicked centre point.

    Click anywhere on the left panel to choose the centre; the right panel
    updates immediately.  Click again to move the centre.  Arrow keys nudge
    the centre by one pixel.

    Requires ``%matplotlib widget`` (ipympl) in the calling notebook cell.

    Parameters
    ----------
    img        : 2D np.array
    cmap       : str
    pixel_size : float or None
        Physical size of one pixel in nm.  When provided the profile x-axis
        is shown in nm; otherwise in pixels.

    Returns
    -------
    result : dict with keys 'center', 'radii', 'profile'
        'radii' are in nm when pixel_size is given, otherwise pixels.
        Values are None until the first click, then updated on each click.
    """
    print("pick_and_profile ready.")
    print("  Click the image to set the centre point.")
    print("  Arrow keys nudge the centre by 1 pixel.")
    print("  Press 's' to save the current profile.")
    print("  Profiles accumulate in result['saved'].")
    print("  When done: radii, profiles = da.stack_profiles(result['saved'])")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Prevent matplotlib's default 's' → save-figure dialog
    _keymap_save = plt.rcParams['keymap.save'][:]
    plt.rcParams['keymap.save'] = [k for k in _keymap_save if k != 's']

    def _restore_keymaps(_):
        plt.rcParams['keymap.save'][:] = _keymap_save
    fig.canvas.mpl_connect('close_event', _restore_keymaps)

    axes[0].imshow(img, cmap=cmap, origin='lower')
    axes[0].set_title('Click to select centre')
    axes[0].axis('off')

    r_label = 'Radius (nm)' if pixel_size is not None else 'Radius (px)'
    axes[1].set_xlabel(r_label)
    axes[1].set_ylabel('Mean intensity')
    axes[1].set_title('Radial profile')
    plt.tight_layout()

    result = {'center': None, 'radii': None, 'profile': None, 'saved': []}

    ny, nx = img.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    def _update(cx, cy):
        max_radius = min(cx, cy, nx - cx, ny - cy)
        if max_radius < 1:
            return

        r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)
        radii = np.arange(0, max_radius)
        profile = np.array([img[r == rad].mean() for rad in radii])
        plot_radii = radii * pixel_size if pixel_size is not None else radii
        result.update(center=(cx, cy), radii=plot_radii, profile=profile)

        n_saved = len(result['saved'])
        save_hint = f'  [{n_saved} saved]' if n_saved else '  [s = save]'

        axes[0].clear()
        axes[0].imshow(img, cmap=cmap, origin='lower')
        for i, s in enumerate(result['saved']):
            sx, sy = s['center']
            axes[0].plot(sx, sy, 'c+', markersize=10, markeredgewidth=1.5)
            axes[0].annotate(str(i + 1), (sx, sy),
                             xytext=(4, 4), textcoords='offset points',
                             color='cyan', fontsize=9, fontweight='bold')
        axes[0].plot(cx, cy, 'w+', markersize=12, markeredgewidth=2)
        axes[0].add_patch(plt.Circle((cx, cy), max_radius, color='white',
                                     fill=False, linewidth=1, linestyle='--'))
        axes[0].set_title(f'Centre: ({cx}, {cy}){save_hint}')
        axes[0].axis('off')

        axes[1].clear()
        for i, s in enumerate(result['saved']):
            axes[1].plot(s['radii'], s['profile'], alpha=0.4,
                         label=str(i + 1))
        axes[1].plot(plot_radii, profile)
        axes[1].set_xlabel(r_label)
        axes[1].set_ylabel('Mean intensity')
        axes[1].set_title('Radial profile')
        if result['saved']:
            axes[1].legend(title='#', fontsize=8)
        fig.canvas.draw_idle()

    def _save():
        if result['center'] is None:
            return
        entry = {'center': result['center'],
                 'radii':  result['radii'].copy(),
                 'profile': result['profile'].copy()}
        result['saved'].append(entry)
        n = len(result['saved'])
        cx, cy = result['center']
        print(f"Saved profile {n}: centre ({cx}, {cy}), "
              f"{len(result['radii'])} points")
        _update(cx, cy)

    def on_click(event):
        if event.inaxes is not axes[0]:
            return
        _update(int(round(event.xdata)), int(round(event.ydata)))

    _arrow = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}

    def on_key(event):
        if result['center'] is None:
            return
        if event.key in _arrow:
            cx, cy = result['center']
            dx, dy = _arrow[event.key]
            _update(cx + dx, cy + dy)
        elif event.key == 's':
            _save()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return result


def _refine_center_gaussian(img, cx, cy, search_radius=20):
    """
    Refine a click position to the centre of the nearest protrusion by fitting
    a 2D Gaussian to a local crop.  Falls back to the local maximum if the fit
    fails or produces an out-of-bounds result.

    Returns refined (cx, cy) in image pixel coordinates.
    """
    ny, nx = img.shape
    r = search_radius
    x0, x1 = max(0, cx - r), min(nx, cx + r + 1)
    y0, y1 = max(0, cy - r), min(ny, cy + r + 1)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape

    xg, yg = np.arange(cw), np.arange(ch)
    xx, yy = np.meshgrid(xg, yg)

    def gaussian_2d(xy, amp, gx, gy, sx, sy, bg):
        x, y = xy
        return (bg + amp * np.exp(
            -((x - gx)**2 / (2 * sx**2) + (y - gy)**2 / (2 * sy**2))
        )).ravel()

    bg_est = crop.min()
    amp_est = crop.max() - bg_est
    p0 = [amp_est, cw / 2, ch / 2, cw / 4, ch / 4, bg_est]
    bounds = ([0, 0, 0, 0.5, 0.5, -np.inf],
              [np.inf, cw, ch, cw, ch, np.inf])

    try:
        popt, _ = curve_fit(gaussian_2d, (xx.ravel(), yy.ravel()),
                            crop.ravel(), p0=p0, bounds=bounds, maxfev=2000)
        refined_cx = int(round(x0 + popt[1]))
        refined_cy = int(round(y0 + popt[2]))
        if 0 <= refined_cx < nx and 0 <= refined_cy < ny:
            return refined_cx, refined_cy
    except Exception:
        pass

    peak = np.unravel_index(crop.argmax(), crop.shape)
    return int(x0 + peak[1]), int(y0 + peak[0])


def pick_and_profile_auto(img, cmap='gist_heat', search_radius=20, pixel_size=None):
    """
    Like pick_and_profile, but refines the clicked position to the centre of
    the nearest protrusion using a 2D Gaussian fit on a local crop.

    Click near a protrusion; the centre is refined automatically and the radial
    profile is computed from the fitted position.  Click again to move.  Arrow
    keys nudge the centre by one pixel.

    Requires ``%matplotlib widget`` (ipympl) in the calling notebook cell.

    Parameters
    ----------
    img           : 2D np.array
    cmap          : str
    search_radius : int    — half-width of the local crop for the Gaussian fit,
                            in pixels (default 20)
    pixel_size    : float or None
        Physical size of one pixel in nm.  When provided the profile x-axis
        is shown in nm; otherwise in pixels.

    Returns
    -------
    result : dict with keys 'click', 'center', 'radii', 'profile'
        'click'  — raw pixel coordinates of the mouse click
        'center' — refined centre from the Gaussian fit (or local max fallback)
        'radii'  — in nm when pixel_size is given, otherwise pixels
        'profile' — mean intensity at each radius
    """
    print("pick_and_profile_auto ready.")
    print("  Click near a protrusion — centre is refined automatically.")
    print("  Arrow keys nudge the centre by 1 pixel.")
    print("  Press 's' to save the current profile.")
    print("  Profiles accumulate in result['saved'].")
    print("  When done: radii, profiles = da.stack_profiles(result['saved'])")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Prevent matplotlib's default 's' → save-figure dialog
    _keymap_save = plt.rcParams['keymap.save'][:]
    plt.rcParams['keymap.save'] = [k for k in _keymap_save if k != 's']

    def _restore_keymaps(_):
        plt.rcParams['keymap.save'][:] = _keymap_save
    fig.canvas.mpl_connect('close_event', _restore_keymaps)

    axes[0].imshow(img, cmap=cmap, origin='lower')
    axes[0].set_title('Click near a protrusion')
    axes[0].axis('off')

    r_label = 'Radius (nm)' if pixel_size is not None else 'Radius (px)'
    axes[1].set_xlabel(r_label)
    axes[1].set_ylabel('Mean intensity')
    axes[1].set_title('Radial profile')
    plt.tight_layout()

    result = {'click': None, 'center': None, 'radii': None, 'profile': None,
              'saved': []}

    ny, nx = img.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    def _update(cx, cy):
        max_radius = min(cx, cy, nx - cx, ny - cy)
        if max_radius < 1:
            return

        r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)
        radii = np.arange(0, max_radius)
        profile = np.array([img[r == rad].mean() for rad in radii])
        plot_radii = radii * pixel_size if pixel_size is not None else radii
        result.update(center=(cx, cy), radii=plot_radii, profile=profile)

        click_cx, click_cy = result['click'] or (cx, cy)
        offset = (cx - click_cx, cy - click_cy)
        n_saved = len(result['saved'])
        save_hint = f'  [{n_saved} saved]' if n_saved else '  [s = save]'

        axes[0].clear()
        axes[0].imshow(img, cmap=cmap, origin='lower')
        for i, s in enumerate(result['saved']):
            sx, sy = s['center']
            axes[0].plot(sx, sy, 'c+', markersize=10, markeredgewidth=1.5)
            axes[0].annotate(str(i + 1), (sx, sy),
                             xytext=(4, 4), textcoords='offset points',
                             color='cyan', fontsize=9, fontweight='bold')
        if result['click'] and result['click'] != (cx, cy):
            axes[0].plot(click_cx, click_cy, 'w+', markersize=10,
                         markeredgewidth=1, alpha=0.5)
        axes[0].plot(cx, cy, 'w+', markersize=12, markeredgewidth=2)
        axes[0].add_patch(plt.Circle((cx, cy), max_radius, color='white',
                                     fill=False, linewidth=1, linestyle='--'))
        axes[0].set_title(
            f'Centre: ({cx}, {cy})  |  offset: {offset}{save_hint}')
        axes[0].axis('off')

        axes[1].clear()
        for i, s in enumerate(result['saved']):
            axes[1].plot(s['radii'], s['profile'], alpha=0.4,
                         label=str(i + 1))
        axes[1].plot(plot_radii, profile)
        axes[1].set_xlabel(r_label)
        axes[1].set_ylabel('Mean intensity')
        axes[1].set_title('Radial profile')
        if result['saved']:
            axes[1].legend(title='#', fontsize=8)
        fig.canvas.draw_idle()

    def _save():
        if result['center'] is None:
            return
        entry = {'center': result['center'],
                 'radii':  result['radii'].copy(),
                 'profile': result['profile'].copy()}
        result['saved'].append(entry)
        n = len(result['saved'])
        cx, cy = result['center']
        print(f"Saved profile {n}: centre ({cx}, {cy}), "
              f"{len(result['radii'])} points")
        _update(cx, cy)

    def on_click(event):
        if event.inaxes is not axes[0]:
            return
        click_cx, click_cy = int(round(event.xdata)), int(round(event.ydata))
        cx, cy = _refine_center_gaussian(img, click_cx, click_cy, search_radius)
        result['click'] = (click_cx, click_cy)
        _update(cx, cy)

    _arrow = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}

    def on_key(event):
        if result['center'] is None:
            return
        if event.key in _arrow:
            cx, cy = result['center']
            dx, dy = _arrow[event.key]
            _update(cx + dx, cy + dy)
        elif event.key == 's':
            _save()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return result


def stack_profiles(saved):
    """
    Stack a list of saved radial profiles into NaN-padded 2D numpy arrays.

    Profiles from different centre points may have different lengths.  Shorter
    profiles are padded with NaN so all rows have the same length.

    Parameters
    ----------
    saved : list of dicts
        Each dict must have keys 'radii' and 'profile' (as returned in
        result['saved'] by pick_and_profile / pick_and_profile_auto).

    Returns
    -------
    radii    : np.ndarray, shape (n, max_len)  — NaN-padded radii
    profiles : np.ndarray, shape (n, max_len)  — NaN-padded intensities
    """
    if not saved:
        return np.array([]), np.array([])
    n = len(saved)
    max_len = max(len(s['radii']) for s in saved)
    radii    = np.full((n, max_len), np.nan)
    profiles = np.full((n, max_len), np.nan)
    for i, s in enumerate(saved):
        L = len(s['radii'])
        radii[i, :L]    = s['radii']
        profiles[i, :L] = s['profile']
    return radii, profiles


def plot_profiles(radii, profiles, pixel_size=None, title='Radial profiles',
                  show_mean=True, crop=True):
    """
    Plot a set of stacked radial profiles as returned by stack_profiles().

    Parameters
    ----------
    radii      : np.ndarray, shape (n, max_len)  — from stack_profiles()
    profiles   : np.ndarray, shape (n, max_len)  — from stack_profiles()
    pixel_size : float or None
        Pass only if radii are still in pixels and you want to convert to nm
        here.  If pixel_size was already supplied when picking, leave as None.
    title      : str
    show_mean  : bool
        Overlay the mean ± std across all profiles (default True).
    crop : bool, optional
        When True (default) the mean and ±1 std band are computed and plotted
        only up to the length of the shortest individual profile, avoiding
        step artefacts caused by NaN-padding.  Individual profile lines are
        always drawn at their full length regardless of this setting.
        When False the NaN-aware mean is computed over the full padded length.

    Returns
    -------
    fig, ax
    """
    radii    = np.atleast_2d(radii)
    profiles = np.atleast_2d(profiles)

    if pixel_size is not None:
        radii = radii * pixel_size
        r_label = 'Radius (nm)'
    else:
        r_label = 'Radius (nm)' if np.nanmax(radii) < 50 else 'Radius (px)'

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (r_row, p_row) in enumerate(zip(radii, profiles)):
        ax.plot(r_row, p_row, alpha=0.4, label=str(i + 1))

    if show_mean and len(profiles) > 1:
        if crop:
            # Crop to the shortest profile so the mean contains no NaN artefacts
            min_len = min(int(np.sum(~np.isnan(p))) for p in profiles)
            r_stat    = np.nanmean(radii[:, :min_len],    axis=0)
            p_mean    = np.nanmean(profiles[:, :min_len], axis=0)
            p_std     = np.nanstd( profiles[:, :min_len], axis=0)
        else:
            r_stat = np.nanmean(radii,    axis=0)
            p_mean = np.nanmean(profiles, axis=0)
            p_std  = np.nanstd( profiles, axis=0)
        ax.plot(r_stat, p_mean, color='black', linewidth=2, label='mean')
        ax.fill_between(r_stat, p_mean - p_std, p_mean + p_std,
                        color='black', alpha=0.15, label='±1 std')

    ax.set_xlabel(r_label)
    ax.set_ylabel('Mean intensity')
    ax.set_title(title)
    ax.legend(title='#', fontsize=8)
    plt.tight_layout()
    plt.show()
    return fig, ax


def save_profiles(radii, profiles, filepath=None, filename=None):
    """
    Save stacked radial profiles to a tab-delimited text file.

    The file has a single shared radius column followed by one column per
    individual profile, then mean and std columns.  Profiles shorter than the
    longest are left blank beyond their last point.  Three ``#``-prefixed
    comment lines precede the column-header row so that all of Excel, Origin,
    Igor Pro, MATLAB, and Python can import the file with minimal configuration.

    Parameters
    ----------
    radii    : np.ndarray, shape (n, max_len)  — from stack_profiles()
    profiles : np.ndarray, shape (n, max_len)  — from stack_profiles()
    filepath : str or None
        Either a directory path or a complete file path.
        - Directory path (or None): the file is placed there; the name comes
          from ``filename`` or is auto-generated with a timestamp.
        - Complete file path (has a file extension / is not an existing
          directory): used as-is; ``filename`` is ignored.
    filename : str or None
        Just the filename (e.g. ``'my_data.txt'``).  Used when ``filepath``
        is a directory or None.  If also None, an inline prompt asks for a
        name; pressing Enter accepts a timestamped default.

    Returns
    -------
    save_path : str  — absolute path actually written.

    Notes
    -----
    Import hints:
      Python  : ``pd.read_csv(path, sep='\\t', comment='#')``
      MATLAB  : ``readtable(path, 'FileType','text', 'Delimiter','\\t', 'CommentStyle','#')``
      Igor Pro: Data > Load Waves > General Text (skip lines starting with #)
      Origin  : Import > Single ASCII, set comment character to #
      Excel   : Data > From Text/CSV, delimiter = Tab
    """
    import datetime

    radii    = np.atleast_2d(radii)
    profiles = np.atleast_2d(profiles)
    n, max_len = profiles.shape

    lengths = [int(np.sum(~np.isnan(p))) for p in profiles]
    min_len = min(lengths)
    longest_idx = int(np.argmax(lengths))

    r_shared = radii[longest_idx].copy()

    r_mean = np.nanmean(radii[:, :min_len],    axis=0)
    p_mean = np.nanmean(profiles[:, :min_len], axis=0)
    p_std  = np.nanstd( profiles[:, :min_len], axis=0)

    # Resolve save_path from filepath + filename
    if filepath is not None and not os.path.isdir(filepath):
        # Treat as a complete file path; filename is ignored
        save_path = filepath
    else:
        # filepath is None or a directory
        save_dir = filepath if filepath is not None else os.getcwd()
        if filename is None:
            default_name = ('radial_profiles_'
                            + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            + '.txt')
            try:
                response = input(f'Filename [{default_name}]: ').strip()
            except EOFError:
                response = ''
            filename = response if response else default_name
            if not response:
                print(f"No name given — using default: {filename}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    col_names = (['radius']
                 + [f'profile_{i + 1}' for i in range(n)]
                 + ['mean', 'std'])

    def _fmt(v):
        return '' if np.isnan(v) else f'{v:.6g}'

    with open(save_path, 'w') as f:
        f.write(f'# Radial profiles  |  saved: {timestamp}\n')
        f.write(f'# n_profiles: {n}  |  max_length: {max_len}'
                f'  |  mean_std_length: {min_len}\n')
        f.write('# blank cells = profile did not extend to this radius\n')
        f.write('\t'.join(col_names) + '\n')
        for k in range(max_len):
            row = [_fmt(r_shared[k])]
            for i in range(n):
                row.append(_fmt(profiles[i, k]))
            row.append(_fmt(p_mean[k]) if k < min_len else '')
            row.append(_fmt(p_std[k])  if k < min_len else '')
            f.write('\t'.join(row) + '\n')

    abs_path = os.path.abspath(save_path)
    print(f"Saved {n} profiles ({max_len} rows)")
    print(f"  Directory : {os.path.dirname(abs_path)}")
    print(f"  File      : {os.path.basename(abs_path)}")
    return abs_path


def radial_profile(img, center, cmap='gist_heat', pixel_size=None):
    """
    Compute and display a radially averaged profile from a given centre point.

    Parameters
    ----------
    img        : 2D np.array
    center     : tuple (x, y) in pixel coordinates
    cmap       : str
    pixel_size : float or None
        Physical size of one pixel in nm.  When provided the profile x-axis
        is shown in nm; otherwise in pixels.

    Returns
    -------
    radii   : np.array  radial distances in nm (or pixels if pixel_size is None)
    profile : np.array  mean intensity at each radius
    """
    cx, cy = center

    ny, nx = img.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)

    max_radius = min(cx, cy, nx - cx, ny - cy)
    radii = np.arange(0, max_radius)
    profile = np.array([img[r == rad].mean() for rad in radii])
    plot_radii = radii * pixel_size if pixel_size is not None else radii

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img, cmap=cmap, origin='lower')
    axes[0].plot(cx, cy, 'w+', markersize=12, markeredgewidth=2)
    axes[0].add_patch(plt.Circle((cx, cy), max_radius, color='white', fill=False,
                                  linewidth=1, linestyle='--'))
    axes[0].set_title(f'Centre: ({cx}, {cy})')
    axes[0].axis('off')

    axes[1].plot(plot_radii, profile)
    axes[1].set_xlabel('Radius (nm)' if pixel_size is not None else 'Radius (px)')
    axes[1].set_ylabel('Mean intensity')
    axes[1].set_title('Radial profile')
    plt.tight_layout()
    plt.show()

    return plot_radii, profile