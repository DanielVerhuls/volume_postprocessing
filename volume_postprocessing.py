from re import S
import tkinter as tk
from tkinter import filedialog
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import numpy as np

class CSVLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV File Loader")

        self.data = None  # 2D array to store CSV data
        self.t_rr = 0
        self.t_clip = 0
        self.time_values = []
        self.volume_values = []
        self.timestep_size = 1

        # Create UI components
        self.btn_load = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(pady=20)

        # Create a matplotlib figure and axis for the plot
        self.figure, self.axis = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

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
                if row_index == 12: self.t_clip = float(row[1])
                if row_index == 13: self.t_rr = float(row[1])
                if row_index == 748 : 
                    for val in row[1:]:
                        if val: self.time_values.append(float(val))
                    self.timestep_size = self.time_values[1] - self.time_values[0]
                elif row_index == 749:
                    for val in row[1:]: 
                        if val: self.volume_values.append(float(val)) 

        self.interpolate_values(target_time_step=1) 
        #if self.t_clip != self.t_rr: self.close_volume_values()
        #self.volume_shift()

    def close_volume_values(self):
        """Linearly interpolate between the last and first volume value if not the whole rr-duration is captured"""
        # Check if the lengths of time_data and volume_data are the same
        if len(self.time_values) != len(self.volume_values):
            raise ValueError("Lengths of time_data and volume_data must be the same.")
        # Calculate linear slope
        slope = (self.volume_values[0] - self.volume_values[-1]) / (self.time_values[0] + self.t_rr - self.time_values[-1])
        # Calculate the amount of missing values
        n_missing = math.floor((self.t_rr - self.t_clip) / self.timestep_size)
        # Fill in missing values for the next cycle
        for i in range(n_missing): # !!! vielleicht doppelt value beim shift
            next_time = self.time_values[-1] + self.timestep_size
            next_volume = self.volume_values[-1] + slope * self.timestep_size
            self.time_values.append(next_time)
            self.volume_values.append(next_volume)

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
        interpolated_time = np.arange(self.time_values[0], self.time_values[-1], target_time_step)
        # Interpolate volume values based on the new time array
        interpolated_volume = np.interp(interpolated_time, self.time_values, self.volume_values)
        self.time_values = interpolated_time.tolist()
        self.volume_values = interpolated_volume.tolist()
        self.timestep_size = target_time_step


    def compute_vol_derivation(self, x, y):
        """Compute derivation"""
        pass

    def plot_data(self):
        """Plot time and volume"""
        if self.data:
            # Clear previous plot
            self.axis.clear()

            # Plot time and volume
            self.axis.plot(self.time_values, self.volume_values, label='Volume vs Time')

            # Set plot labels and legend
            self.axis.set_xlabel('Time (ms)')
            # Set plot labels and legend
            self.axis.set_ylabel('Volume (ml)')
            self.axis.legend()

            # Update canvas
            self.canvas.draw()


# Create the main application window
root = tk.Tk()
app = CSVLoaderApp(root)

# Start the main event loop
root.mainloop()
