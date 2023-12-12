from re import S
import tkinter as tk
from tkinter import filedialog
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

class CSVLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Volume post-processing")

        self.data = None  # 2D array to store CSV data
        self.t_rr = 0
        self.t_clip = 0
        self.time_values = []
        self.volume_values = []
        self.d_vol_dt = []
        self.d_d_vol_dt_dt = []
        self.timestep_size = 1 # 1 ms
        # Export values
        self.EDV = 0
        self.ESV = 0
        self.PER = 0
        self.PFR = 0
        self.time_to_PER = 0
        self.time_to_PFR = 0

        # Create UI components
        self.btn_load = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(pady=20)

        # Create a matplotlib figure and axis for the volume plot
        self.figure1, self.axis1 = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=root)
        self.canvas_widget1 = self.canvas1.get_tk_widget()
        self.canvas_widget1.pack()

        # Create a matplotlib figure and axis for the second plot 
        self.figure2, self.axis2 = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=root)
        self.canvas_widget2 = self.canvas2.get_tk_widget()
        self.canvas_widget2.pack()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            # Clear existing data
            self.data = []
            # Read CSV file and store data in the 2D array
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                for row in csv_reader:
                    # Convert each cell value to float and add to the 2D array
                    row_data = [val for val in row]
                    self.data.append(row_data)
        self.get_time_and_vol()
        self.plot_data()

    def get_time_and_vol(self):
        """Read time and volume data"""
        if self.data:
            # Read time and volume data
            self.time_values = []
            self.volume_values = []
            for row_index, row in enumerate(self.data):
                if row_index == 13: self.t_rr = float(row[1])
                elif row_index == 748 : 
                    for val in row[1:]:
                        if val: self.time_values.append(float(val))
                    self.timestep_size = self.time_values[1] - self.time_values[0]
                    self.t_clip = max(self.time_values)
                elif row_index == 749:
                    for val in row[1:]: 
                        if val: self.volume_values.append(float(val)) 
        # Increase temporal resolution through interpolation
        self.interpolate_values(target_time_step=1) 
        # Close the gap of the recorded data (e.g. if only a part of t_rr has been captured)
        if self.t_clip != self.t_rr: self.close_volume_values()
        # Shift the volumes such that the EDV is at the begining
        self.volume_shift()
        # Apply smoothing filter to volumes
        self.savitzky_golay_filter(case="vol")
        # Compute volume derivations
        self.compute_vol_derivations()
        

        # Create a mask to specify different regions
        mask = np.zeros((3, len(self.volume_values)))  # Two regions
        mask[0, 0:150] = 1  # Apply filter to the first region
        mask[1, 150:500] = 1    # Apply filter to the second region
        mask[1, 500:] = 1    # Apply filter to the third region

        # Define window sizes and orders for each region
        window_sizes = [3, 31, 3]
        orders = [2, 4, 2]

        self.d_vol_dt = self.apply_variable_strength_savitzky_golay(self.d_vol_dt, window_sizes, orders, mask)
        print(f"Derivative volume value at index 300: {self.d_vol_dt[300]}")
        print(f"Derivative minimum volume: {min(self.d_vol_dt)}")
        print(f"Derivative maximum volume: {max(self.d_vol_dt)}")
        #self.savitzky_golay_filter(case="d_vol_dt")
        print(f"Derivative volume value at index 300: {self.d_vol_dt[300]}")
        print(f"Derivative minimum volume: {min(self.d_vol_dt)}")
        print(f"Derivative maximum volume: {max(self.d_vol_dt)}")
        self.compute_exports()
        
    def close_volume_values(self):
        """Linearly interpolate between the last and first volume value if not the whole rr-duration is captured"""
        # Check if the lengths of time_data and volume_data are the same
        if len(self.time_values) != len(self.volume_values):
            raise ValueError("Lengths of time_data and volume_data must be the same.")
        # Calculate the amount of missing values
        n_missing = math.floor((self.t_rr - self.t_clip) / self.timestep_size)
        # Convert arrays for Numpy
        times = np.array(self.time_values)
        volumes = np.array(self.volume_values)
        # Append last element as initial element
        times = np.append(times, self.t_rr)
        volumes = np.append(volumes, volumes[0])
        # Compute new time values after spline interpolation
        plot_time_values = np.linspace(min(times), max(times), num=round(self.t_rr)) 
        self.time_values = plot_time_values.tolist()
        # Compute spline of the volumes values
        spline = CubicSpline(times, volumes, bc_type='periodic')
        spline_values = spline(plot_time_values)
        self.volume_values = spline_values.tolist()
        
    def volume_shift(self, ):
        """Shift volumes such that the initial value begins with EDV"""
        if not self.volume_values:
            # Return an empty list if the input list is empty
            return []
        # Find the index of the maximum value in the list
        max_index = self.volume_values.index(max(self.volume_values))
        # Rotate the list to move the maximum value to the beginning
        temp = self.volume_values[max_index:] + self.volume_values[:max_index]
        self.volume_values = temp

    def interpolate_values(self, target_time_step):
        """Interpolate volume-time curve to a specified time step"""
        # Create a new time array with the desired time step
        interpolated_time = np.arange(self.time_values[0], round(self.time_values[-1]), target_time_step)
        # Interpolate volume values based on the new time array
        interpolated_volume = np.interp(interpolated_time, self.time_values, self.volume_values)
        self.time_values = interpolated_time.tolist()
        self.volume_values = interpolated_volume.tolist()
        self.timestep_size = target_time_step
        self.t_clip = max(self.time_values)

    def savitzky_golay_filter(self, case):
        """Apply Savitzky-Golay filter to smooth volume values"""
        if case == "vol":
            smoothed_curve = savgol_filter(self.volume_values, window_length = 5, polyorder = 4)
            self.volume_values = smoothed_curve
        elif case == "d_vol_dt":
            smoothed_curve = savgol_filter(self.d_vol_dt, window_length = 10, polyorder = 4)
            self.d_vol_dt = smoothed_curve
        else:
            print(f"Wrong case for filter.")
            return False
        
    def apply_variable_strength_savitzky_golay(self, volume_curve, window_sizes, orders, mask):
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
            # Throw error for faulty window sizes
            if window_size % 2 == 0 or window_size < 1:
                raise ValueError(f"Invalid window size {window_size} for region {i}")
            # Apply savgol filter to specified region
            smoothed_region = savgol_filter(region_curve, window_size, order)
            smoothed_curve += smoothed_region
        return smoothed_curve

    def compute_vol_derivations(self):
        """Compute derivation"""
        self.d_vol_dt = []
        for i in range(len(self.volume_values)):
            if i == 0: self.d_vol_dt.append((self.volume_values[i] - self.volume_values[-1]) / self.timestep_size)
            else: self.d_vol_dt.append((self.volume_values[i] - self.volume_values[i-1]) / self.timestep_size)
        # !!! zweite ableitung

    def compute_exports(self):
        """Compute exports"""
        print(f"Computing exports")
        self.EDV = max(self.volume_values)
        print(f"EDV: {self.EDV} ml")
        self.ESV = min(self.volume_values)
        print(f"ESV: {self.ESV} ml")
        self.PER = max(self.d_vol_dt)
        print(f"PER: {self.PER} l/s")
        self.PFR = min(self.d_vol_dt)
        print(f"PFR: {self.PFR} l/s")
        self.time_to_PER = self.d_vol_dt.tolist().index(min(self.d_vol_dt))
        print(f"time_to_PER: {self.time_to_PER} ms")
        self.time_to_PFR = self.d_vol_dt.tolist().index(max(self.d_vol_dt))
        print(f"time_to_PFR: {self.time_to_PFR} ms")        

    def plot_data(self):
        """Plot time and volume"""
        if self.data:
            ## Plot time and volume
            self.axis1.clear() # Clear previous plot
            self.axis1.plot(self.time_values, self.volume_values, label='Volume vs Time')
            # Set plot labels and legend
            self.axis1.set_xlabel('Time (ms)')
            self.axis1.set_ylabel('Volume (ml)')
            self.axis1.legend()
            self.canvas1.draw() # Update canvas
            
            ## Plot derivation
            self.axis2.clear() # Clear previous plot
            self.axis2.plot(self.time_values, self.d_vol_dt, label='Volume derivation vs Time')
            # Set plot labels and legend
            self.axis2.set_xlabel('Time (ms)')
            self.axis2.set_ylabel('Volume change (ml/s)')
            self.axis2.legend()
            self.canvas2.draw() # Update canvas for the second plot


# Create the main application window
root = tk.Tk()
app = CSVLoaderApp(root)

# Start the main event loop
root.mainloop()
