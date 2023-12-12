import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt  # Optional, for visualization

def apply_variable_strength_savitzky_golay(volume_curve, window_sizes, orders, mask):
    """
    Apply variable-strength Savitzky-Golay filter to a volume-time curve.

    Parameters:
    - volume_curve (numpy.ndarray): Input volume-time curve.
    - window_sizes (list): List of window sizes for each region in the mask.
    - orders (list): List of orders for each region in the mask.
    - mask (numpy.ndarray): Binary mask indicating regions to apply different filter strengths.

    Returns:
    - smoothed_curve (numpy.ndarray): Volume-time curve after applying the variable-strength Savitzky-Golay filter.
    """

    # Ensure the lengths of window_sizes, orders, and mask match
    if len(window_sizes) != len(orders) or len(window_sizes) != mask.shape[0]:
        raise ValueError("Lengths of window_sizes, orders, and mask must match")

    # Apply variable-strength Savitzky-Golay filter
    smoothed_curve = np.zeros_like(volume_curve, dtype=np.float64)

    for i in range(len(window_sizes)):
        region_mask = mask[i, :]
        region_curve = volume_curve * region_mask
        window_size = window_sizes[i]
        order = orders[i]

        if window_size % 2 == 0 or window_size < 1:
            raise ValueError(f"Invalid window size {window_size} for region {i}")

        smoothed_region = savgol_filter(region_curve, window_size, order)
        smoothed_curve += smoothed_region

    return smoothed_curve

# Example usage:
# Generate a sample volume-time curve
time = np.linspace(0, 10, 100)
volume_curve = 2 * np.sin(time) + np.random.normal(scale=0.2, size=len(time))

# Create a mask to specify different regions
mask = np.zeros((2, len(time)))  # Two regions
mask[0, 20:50] = 1  # Apply filter to the first region
mask[1, 70:] = 1    # Apply filter to the second region

# Define window sizes and orders for each region
window_sizes = [5, 9]
orders = [3, 5]

# Apply the variable-strength Savitzky-Golay filter
smoothed_curve = apply_variable_strength_savitzky_golay(volume_curve, window_sizes, orders, mask)

# Plot the original and smoothed curves for visualization
plt.plot(time, volume_curve, label='Original Curve')
plt.plot(time, smoothed_curve, label='Smoothed Curve', linestyle='--', linewidth=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Volume')
plt.title('Volume-Time Curve with Variable-Strength Savitzky-Golay Smoothing')
plt.show()
