import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# 1. Load the binary data
GRID_SIZE = 50
# X: Input (Boundary conditions + Source)
# Y: Target (Ground Truth solution from C engine)
X_data = np.fromfile("x_train.bin", dtype=np.float32).reshape(-1, GRID_SIZE, GRID_SIZE)
Y_data = np.fromfile("y_train.bin", dtype=np.float32).reshape(-1, GRID_SIZE, GRID_SIZE)

class DataBrowser:
    def __init__(self):
        self.index = 0
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.2)  # Create space for buttons at the bottom

        # Initialize colorbars as None
        self.cbar1 = None
        self.cbar2 = None
        
        self.update_plot()

    def update_plot(self):
        self.axes[0].clear()
        self.axes[1].clear()

        # Display Input (Constraints)
        img1 = self.axes[0].imshow(X_data[self.index], cmap='inferno', vmin=0, vmax=100)
        self.axes[0].set_title(f"Input Grid (Sample: {self.index})")
        
        # Display Target (Physics Solution)
        img2 = self.axes[1].imshow(Y_data[self.index], cmap='inferno', vmin=0, vmax=100)
        self.axes[1].set_title("Target Solution (C-Engine)")

        # Add colorbars only once
        if self.cbar1 is None:
            self.cbar1 = self.fig.colorbar(img1, ax=self.axes[0], fraction=0.046, pad=0.04)
            self.cbar2 = self.fig.colorbar(img2, ax=self.axes[1], fraction=0.046, pad=0.04)
            self.cbar1.set_label('Potential (V)')

        self.fig.canvas.draw_idle()

    def show_next(self, event):
        self.index = (self.index + 1) % len(X_data)
        self.update_plot()

    def show_prev(self, event):
        self.index = (self.index - 1) % len(X_data)
        self.update_plot()

# Initialize the browser object
browser = DataBrowser()

# Button Positions: [left, bottom, width, height]
ax_prev = plt.axes([0.70, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])

button_next = Button(ax_next, 'Next ->')
button_prev = Button(ax_prev, '<- Prev')

# Connect buttons to functions
button_next.on_clicked(browser.show_next)
button_prev.on_clicked(browser.show_prev)

plt.show()