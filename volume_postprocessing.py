import tkinter as tk
from tkinter import filedialog
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CSVLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV File Loader")

        self.data = None  # 2D array to store CSV data

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
        self.plot_volume()

    def plot_volume(self):
        """Plot time and volume"""
        if self.data:
            # Assume each row has the same number of columns
            num_columns = len(self.data[0])
            for row_index, row in enumerate(self.data):
                if row_index == 748 : 
                    time_values = []
                    for val in row[1:]:
                        if val: time_values.append(float(val))
                elif row_index == 749:
                    volume_values = []
                    for val in row[1:]: 
                        if val: volume_values.append(float(val))

            # Clear previous plot
            self.axis.clear()

            # Plot time and volume
            self.axis.plot(time_values, volume_values, label='Volume vs Time')

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
