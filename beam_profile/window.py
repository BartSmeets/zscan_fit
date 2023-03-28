###########################
# IMPORTS
###########################

# Python standard library
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os

from main import main

###########################
# WINDOW
###########################

# Set initial path
INITIAL_PATH = os.environ.get('HOMEPATH')

# Create Window
root = tk.Tk()
root.geometry('500x200')
root.title('Beam Profile')
root.resizable(0,0)
root.attributes('-topmost', True)

# File name directory
## Set working directory
INITIAL_PATH = os.environ.get('HOMEPATH')
## Define browse function to assign to button
def browse(directory):
    file.set(filedialog.askdirectory(initialdir=INITIAL_PATH, title='Select directory'))
    show.set(file.get())
    return  
## Create storage object and assign initial value
file = tk.StringVar(value='')
show = tk.StringVar(value='')
## Create and place labels
file_label = ttk.Label(root, text='Data directory:')
file_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
## Create and place entry field
file_entry = tk.Entry(root, textvariable=show, width=50)
file_entry.grid(column=1, row=0, columnspan=2, sticky=tk.E, padx=5, pady=5)
## Create and place browse button
file_browse= tk.Button(root, text='Browse', command=lambda: browse(INITIAL_PATH))
file_browse.grid(column=3, row=0, sticky=tk.E, padx=5, pady=5)

# Wavelength
## Create storage object and assign initial value
wavelength = tk.StringVar(value = str(532))
## Create labels
wavelength_label = ttk.Label(root, text=r'Wavelength:')
wavelength_unit_label = ttk.Label(root, text='nm')
## Place labels in window
wavelength_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
wavelength_unit_label.grid(column=3, row=1, sticky=tk.W, padx=5, pady=5)
## Create and place entry field
wavelength_entry = tk.Entry(root, textvariable=wavelength)
wavelength_entry.grid(column=1, row=1, columnspan=2, sticky=tk.E, padx=5, pady=5)

# Step size
## Create storage object and assign initial value
step_size = tk.StringVar(value = str(1))
## Create labels
step_size_label = ttk.Label(root, text=r'Step size:')
step_size_unit_label = ttk.Label(root, text='mm')
## Place labels in window
step_size_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
step_size_unit_label.grid(column=3, row=2, sticky=tk.W, padx=5, pady=5)
## Create and place entry field
w0_entry = tk.Entry(root, textvariable=step_size)
w0_entry.grid(column=1, row=2, columnspan=2, sticky=tk.E, padx=5, pady=5)

# Progress bar
## Create and place labels
pb_label = ttk.Label(root, text='Progress:')
pb_label.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
pb = ttk.Progressbar(root, orient='horizontal', mode='determinate', length='280')
pb.grid(column=1, row=3, columnspan=2, padx=10, pady=20)
# Apply button
run = tk.Button(root, text='Run', command=lambda: main(pb, root, file, wavelength, step_size))
run.grid(column=3, row=3, sticky=tk.E, padx=5, pady=5)

# Close button
close = tk.Button(root, text='Close', command=lambda: root.quit())
close.grid(column=3, row=4, sticky=tk.E, padx=5, pady=5)

root.mainloop()