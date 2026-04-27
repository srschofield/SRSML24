


def radial_profile(img, center=None, cmap='gist_heat'):
    """
    Compute and display a radially averaged line profile from a chosen point.

    Parameters
    ----------
    img : 2D np.array
        Image data.
    center : tuple (x, y), optional
        Centre point in pixel coordinates. If None, prompts the user
        to click a point on the image.
    cmap : str
        Colormap for the image display.

    Returns
    -------
    radii : np.array
        Radial distances in pixels.
    profile : np.array
        Mean intensity at each radius.
    center : tuple
        The (x, y) centre point used.
    """
    # If no centre given, let the user click one
    if center is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap=cmap, origin='lower')
        ax.set_title('Click a point to centre the radial profile')
        points = plt.ginput(1, timeout=0)  # wait for 1 click
        plt.close(fig)
        if not points:
            raise ValueError("No point selected.")
        center = (int(round(points[0][0])), int(round(points[0][1])))
        print(f"Centre selected: x={center[0]}, y={center[1]}")

    cx, cy = center

    # Build a grid of distances from the centre
    ny, nx = img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Bin by integer radius and average
    r_int = r.astype(int)
    max_radius = min(cx, cy, nx - cx, ny - cy)  # stay within image bounds
    radii = np.arange(0, max_radius)
    profile = np.array([img[r_int == rad].mean() for rad in radii])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img, cmap=cmap, origin='lower')
    axes[0].plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
    circle = plt.Circle((cx, cy), max_radius, color='red', fill=False, linewidth=1, linestyle='--')
    axes[0].add_patch(circle)
    axes[0].set_title('Image with centre point')
    axes[0].axis('off')

    axes[1].plot(radii, profile)
    axes[1].set_xlabel('Radius (px)')
    axes[1].set_ylabel('Mean intensity')
    axes[1].set_title('Radial profile')
    plt.tight_layout()
    plt.show()

    return radii, profile, center