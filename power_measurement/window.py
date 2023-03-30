####################################
#IMPORTS
###################################

## Python standard library
from main import main
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os

####################################
#CREATE Windows
####################################

# Set initial path
INITIAL_PATH = os.environ.get('HOMEPATH')

# Create Window
root = tk.Tk()
root.geometry('600x150')
root.title('Power measurement')
root.resizable(0,0)
root.attributes('-topmost', True)

## Define browse function to assign to button
def browse():
    file.set(filedialog.askopenfilename(initialdir=INITIAL_PATH, title='Select Power Data'))
    SHOW = file.get()
    show.set(SHOW[SHOW.rfind('/') + 1 :])    # Cut file-location at last / -> only show file name
    return

## Create storage object and assign initial value
file = tk.StringVar(value='')
show = tk.StringVar(value='')

# Asking for file
## Create labels
file_label = ttk.Label(root, text='Data File:')
file_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
## Create and place file field
file_entry = tk.Entry(root, textvariable=show, width = 75)
file_entry.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)
## Create and place browse button
file_browse= tk.Button(root, text='Browse', command=lambda: browse())
file_browse.grid(column=3, row=0, sticky=tk.E, padx=5, pady=5)

# Progress bar
## Create and place labels
pb_label = ttk.Label(root, text='Progress:')
pb_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
pb = ttk.Progressbar(root, orient='horizontal', mode='determinate', length='280')
pb.grid(column=1, row=1, columnspan=2, padx=10, pady=20)
# Apply button
run = tk.Button(root, text='Run', command=lambda: main(pb, root, file))
run.grid(column=3, row=1, sticky=tk.E, padx=5, pady=5)

# Close button
close = tk.Button(root, text='Close', command=lambda: root.quit())
close.grid(column=3, row=2, sticky=tk.E, padx=5, pady=5)

root.mainloop()