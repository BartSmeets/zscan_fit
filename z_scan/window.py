# Python standard library
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Required
import matplotlib.pyplot as plt

# Included in repository
import errorbar
import export_functions
import fitting_model
import main


############################################################
# Writing variables
############################################################
fig_show = plt.figure()
PARAMETER_DATA = [0, 0, 0, 0, 0, 0]

def browse_button():
    global MEASUREMENT, PARAMETER_DATA, FILE_NAME, DIRECTORY, TITLE, fig1

    # Load data
    MEASUREMENT, PARAMETER_DATA, FILE_NAME, DIRECTORY = main.load(file)

    # Change information in windows
    parameter_var.set(export_functions.text.parameter_string(PARAMETER_DATA))
    result_var.set('Run to generate results')
    parameter_frame.update_idletasks()
    
    # Plot raw data
    fig1 = export_functions.plot.plot_raw(MEASUREMENT)
    canvas = FigureCanvasTkAgg(fig1, master = figure_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)
    return

def run():
    global fig2, fig2_1, fig2_2, fig3, OUTPUT_STRING, TITLE
    # Read Input
    ## First Guess
    Z0_0 = float(z0.get())
    I_S1_0 = float(Is1.get())
    I_S2_0 = float(Is2.get())
    BETA_0 = float(beta.get())
    ## Model Parameters
    [L, ALPHA0, I0, Z_R, _, _] = PARAMETER_DATA
    N_RUNS = int(n_runs.get())
    MAX_PERTURBATION = float(pert.get())
    MAX_AGE = int(age.get())
    MAX_ITER = int(iter.get())
    T = float(t.get())
    MAX_JUMP = int(max_jump.get())
    MAX_REJECT = int(max_reject.get())
    
    ## Convert Bound stringVar to float or None
    bounds = [[z0_bounds_l, z0_bounds_r], 
              [Is1_bounds_l, Is1_bounds_r], 
              [Is2_bounds_l, Is1_bounds_r],
              [beta_bounds_l, beta_bounds_r]]
    BOUNDS = main.bounds_convert(bounds)
    

    ### Fit type
    FIT_TYPE = main.fit_type_convert(fit_type)

    # Pack Parameters
    P0 = [Z0_0, I_S1_0, I_S2_0, BETA_0]
    MODEL_PARAMETERS = [MAX_PERTURBATION, MAX_AGE, MAX_ITER, BOUNDS, T, MAX_JUMP, MAX_REJECT]
    EXPERIMENT_PARAM = [L, ALPHA0, I0, Z_R]

    # Run model
    RUNS, P_BEST, CHI2_BEST = fitting_model.run(MEASUREMENT, FIT_TYPE, N_RUNS, P0, MODEL_PARAMETERS, EXPERIMENT_PARAM, 
                                                pb_frame, iter_count_label, value_label, age_count_label, pb)
    SIGMA_P, CHI2_SPAN = errorbar.compute(MEASUREMENT, FIT_TYPE, P_BEST, CHI2_BEST, EXPERIMENT_PARAM)
    OUTPUT_STRING = export_functions.text.output(FIT_TYPE, CHI2_BEST, P_BEST, SIGMA_P, CHI2_SPAN)

    # Update results in window
    result_var.set(OUTPUT_STRING)
    parameter_frame.update_idletasks()

    # Plots
    TITLE, fig2, fig2_1, fig2_2, fig3 = main.generate_plots(
        MEASUREMENT, FILE_NAME, FIT_TYPE, N_RUNS, P_BEST, EXPERIMENT_PARAM, RUNS)

    # Place figure in window
    canvas = FigureCanvasTkAgg(fig2, master = figure_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=3)

    return

############################################################
# Window
############################################################

window = tk.Tk()
#window.attributes('-topmost', True)
window.title('Z-Scan Fit')
window.state('zoomed')

############################################################
# File selection
############################################################

# Create frame
file_frame = ttk.LabelFrame(window, text='File selection')
file_frame.grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky='news')

# Frame contents
## Label
file_label = ttk.Label(file_frame, text='File: ')
file_label.grid(row=0, column=0, padx=5, pady=5)
## Content
file = tk.StringVar(file_frame)
file_contents = tk.Entry(file_frame, textvariable=file, width=190)
file_contents.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
## Browse Button
file_browse= tk.Button(file_frame, text='Browse', command=lambda: browse_button()) 
file_browse.grid(row=0, column=4, padx=5, pady=5)



############################################################
# Input frame
############################################################

# Create frame
input_frame = ttk.LabelFrame(window, text='User input')
input_frame.grid(row=1, column=0, rowspan=4, padx=20, pady=10, sticky='news')


# Frame contents

## Fit type
### Label
fit_type_label = ttk.Label(input_frame, text='Fit type: ')
fit_type_label.grid(row=0, column=0, padx=5, pady=5)
### Contents
fit_type = tk.StringVar(input_frame, '1PA')
fit_type_contents = ttk.Combobox(input_frame, textvariable=fit_type,
                                 values=['1PA', '2PA', '2PA without Is2', '2PA without Is1', '2PA without saturation'])
fit_type_contents.grid(row=0, column=1, padx=5, pady=5)


## Initial guess
initial_frame = ttk.LabelFrame(input_frame, text='Initial guess')
initial_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=2)
### z0
z0 = tk.StringVar(initial_frame, '0')
z0_label = ttk.Label(initial_frame, text='z0 = ')
z0_content = ttk.Entry(initial_frame, textvariable=z0)
z0_unit_label = ttk.Label(initial_frame, text='mm')
z0_label.grid(row=0, column=0, pady=5, sticky=tk.E)
z0_content.grid(row=0, column=1, pady=5)
z0_unit_label.grid(row=0, column=2, pady=5, sticky=tk.W)
### Is1
Is1 = tk.StringVar(initial_frame, '1e6')
Is1_label = ttk.Label(initial_frame, text='Is1 = ')
Is1_content = ttk.Entry(initial_frame, textvariable=Is1)
Is1_unit_label = ttk.Label(initial_frame, text='W/mm2')
Is1_label.grid(row=1, column=0, pady=5, sticky=tk.E)
Is1_content.grid(row=1, column=1, pady=5)
Is1_unit_label.grid(row=1, column=2, pady=5, sticky=tk.W)
### Is2
Is2 = tk.StringVar(initial_frame, '1e-6')
Is2_label = ttk.Label(initial_frame, text='Is2 = ')
Is2_content = ttk.Entry(initial_frame, textvariable=Is2)
Is2_unit_label = ttk.Label(initial_frame, text='W/mm2')
Is2_label.grid(row=2, column=0, pady=5, sticky=tk.E)
Is2_content.grid(row=2, column=1, pady=5)
Is2_unit_label.grid(row=2, column=2, pady=5, sticky=tk.W)
### Beta
beta = tk.StringVar(initial_frame, '1e-6')
beta_content = ttk.Entry(initial_frame, textvariable=beta)
beta_label = ttk.Label(initial_frame, text='beta = ')
beta_unit_label = ttk.Label(initial_frame, text='mm/W')
beta_label.grid(row=3, column=0, pady=5, sticky=tk.E)
beta_content.grid(row=3, column=1, pady=5)
beta_unit_label.grid(row=3, column=2, pady=5, sticky=tk.W)


## model parameters
model_frame = ttk.LabelFrame(input_frame, text='Model parameters')
model_frame.grid(row=2, column=0, padx=5, pady=5, columnspan=2, rowspan=2, sticky='news')

### Bounds
bounds_frame = ttk.LabelFrame(model_frame, text='Bounds')
bounds_frame.grid(row=0, column=0, padx=5, pady=5, columnspan=2)
#### z0
z0_bounds_l = tk.StringVar(bounds_frame, 'None')
z0_bounds_r = tk.StringVar(bounds_frame, 'None')
z0_bounds_label = ttk.Label(bounds_frame, text='z0: ')
z0_bounds_label_ = ttk.Label(bounds_frame, text='-')
z0_bounds_content_l = ttk.Entry(bounds_frame, textvariable=z0_bounds_l)
z0_bounds_content_r = ttk.Entry(bounds_frame, textvariable=z0_bounds_r)
z0_bounds_label.grid(row=0, column=0, pady=5, sticky=tk.E)
z0_bounds_content_l.grid(row=0, column=1, pady=5)
z0_bounds_label_.grid(row=0, column=2, pady=5)
z0_bounds_content_r.grid(row=0, column=3, pady=5)
#### Is1
Is1_bounds_l = tk.StringVar(bounds_frame, '1e-99')
Is1_bounds_r = tk.StringVar(bounds_frame, 'None')
Is1_bounds_label = ttk.Label(bounds_frame, text='Is1: ')
Is1_bounds_label_ = ttk.Label(bounds_frame, text='-')
Is1_bounds_content_l = ttk.Entry(bounds_frame, textvariable=Is1_bounds_l)
Is1_bounds_content_r = ttk.Entry(bounds_frame, textvariable=Is1_bounds_r)
Is1_bounds_label.grid(row=1, column=0, pady=5, sticky=tk.E)
Is1_bounds_content_l.grid(row=1, column=1, pady=5)
Is1_bounds_label_.grid(row=1, column=2, pady=5)
Is1_bounds_content_r.grid(row=1, column=3, pady=5)
#### Is2
Is2_bounds_l = tk.StringVar(bounds_frame, '1e-99')
Is2_bounds_r = tk.StringVar(bounds_frame, 'None')
Is2_bounds_label = ttk.Label(bounds_frame, text='Is2: ')
Is2_bounds_label_ = ttk.Label(bounds_frame, text='-')
Is2_bounds_content_l = ttk.Entry(bounds_frame, textvariable=Is2_bounds_l)
Is2_bounds_content_r = ttk.Entry(bounds_frame, textvariable=Is2_bounds_r)
Is2_bounds_label.grid(row=2, column=0, pady=5, sticky=tk.E)
Is2_bounds_content_l.grid(row=2, column=1, pady=5)
Is2_bounds_label_.grid(row=2, column=2, pady=5)
Is2_bounds_content_r.grid(row=2, column=3, pady=5)
#### beta
beta_bounds_l = tk.StringVar(bounds_frame, '0')
beta_bounds_r = tk.StringVar(bounds_frame, 'None')
beta_bounds_label = ttk.Label(bounds_frame, text='beta: ')
beta_bounds_label_ = ttk.Label(bounds_frame, text='-')
beta_bounds_content_l = ttk.Entry(bounds_frame, textvariable=beta_bounds_l)
beta_bounds_content_r = ttk.Entry(bounds_frame, textvariable=beta_bounds_r)
beta_bounds_label.grid(row=3, column=0, pady=5, sticky=tk.E)
beta_bounds_content_l.grid(row=3, column=1, pady=5)
beta_bounds_label_.grid(row=3, column=2, pady=5)
beta_bounds_content_r.grid(row=3, column=3, pady=5)

### Model
#### N_RUNS
n_runs = tk.StringVar(model_frame, '5')
n_label = ttk.Label(model_frame, text='N_RUNS = ')
n_content = tk.Entry(model_frame, textvariable=n_runs)
n_label.grid(row=1, column=0, pady=5, sticky=tk.E)
n_content.grid(row=1, column=1, pady=5, sticky=tk.W)
#### MAX PERTURBATION
pert = tk.StringVar(model_frame, '2')
pert_label = ttk.Label(model_frame, text='MAX_PERTURBATION = ')
pert_content = tk.Entry(model_frame, textvariable=pert)
pert_label.grid(row=2, column=0, pady=5, sticky=tk.E)
pert_content.grid(row=2, column=1, pady=5, sticky=tk.W)
#### MAX ITER
iter = tk.StringVar(model_frame, '500')
iter_label = ttk.Label(model_frame, text='MAX_ITER = ')
iter_content = tk.Entry(model_frame, textvariable=iter)
iter_label.grid(row=2, column=0, pady=5, sticky=tk.E)
iter_content.grid(row=2, column=1, pady=5, sticky=tk.W)
#### MAX AGE
age = tk.StringVar(model_frame, '50')
age_label = ttk.Label(model_frame, text='MAX_AGE = ')
age_content = tk.Entry(model_frame, textvariable=age)
age_label.grid(row=3, column=0, pady=5, sticky=tk.E)
age_content.grid(row=3, column=1, pady=5, sticky=tk.W)
#### T
t = tk.StringVar(model_frame, '0.8')
t_label = ttk.Label(model_frame, text='T = ')
t_content = tk.Entry(model_frame, textvariable=t)
t_label.grid(row=4, column=0, pady=5, sticky=tk.E)
t_content.grid(row=4, column=1, pady=5, sticky=tk.W)
#### MAX JUMP
max_jump = tk.StringVar(model_frame, '5')
jump_label = ttk.Label(model_frame, text='MAX_JUMP = ')
jump_content = tk.Entry(model_frame, textvariable=max_jump)
jump_label.grid(row=5, column=0, pady=5, sticky=tk.E)
jump_content.grid(row=5, column=1, pady=5, sticky=tk.W)
#### MAX REJECT
max_reject = tk.StringVar(model_frame, '5')
reject_label = ttk.Label(model_frame, text='MAX_REJECT = ')
reject_content = tk.Entry(model_frame, textvariable=max_reject)
reject_label.grid(row=6, column=0, pady=5, sticky=tk.E)
reject_content.grid(row=6, column=1, pady=5, sticky=tk.W)


############################################################
# Parameters
############################################################

# Create frame
parameter_frame = ttk.LabelFrame(window, text='Experiment parameters')
parameter_frame.grid(row=1, column=1, padx=20, pady=10, sticky='news')

parameter_var = tk.StringVar(parameter_frame, export_functions.text.parameter_string(PARAMETER_DATA))
parameter_content = ttk.Label(parameter_frame, textvariable=parameter_var)
parameter_content.grid(row=0, column=0, padx=5, pady=5)


############################################################
# Results
############################################################

# Create frame
results_frame = ttk.LabelFrame(window, text='Fitting results', width=200)
results_frame.grid(row=2, column=1, padx=20, pady=10, sticky='news')

result_var = tk.StringVar(parameter_frame, 'Load data')
results_str = ttk.Label(results_frame, textvariable=result_var)
results_str.grid(row=0, column=0, padx=5, pady=5)

############################################################
# Progress bar
############################################################
pb_frame = ttk.LabelFrame(window, text='Progress bar')
pb_frame.grid(row=3, column=1, padx=20, pady=10)

pb = ttk.Progressbar(pb_frame, orient='horizontal', mode='determinate', length='280')
pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)
## Initialise progress bar
progress = 'Number of models computed: 0/' + n_runs.get()
value_label = ttk.Label(pb_frame, text=progress)
value_label.grid(column=0, row=1, columnspan=2)
## Iteration number
iter_count_label = ttk.Label(pb_frame, text='Iteration: ' + str(0) + '/' + iter.get())
iter_count_label.grid(column=0, row=2, columnspan=2)
## Age
age_count_label = ttk.Label(pb_frame, text='Best age: ' + str(0) + '/' + age.get())
age_count_label.grid(column=0, row=3, columnspan=2)

############################################################
# Figure
############################################################

figure_frame = ttk.LabelFrame(window, text='Plot')
figure_frame.grid(row=1, column=2, rowspan=3, columnspan=3 ,padx=20, pady=10, sticky='news')

canvas = FigureCanvasTkAgg(fig_show, master = figure_frame)  
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky='news')


# Buttons
T_I_button = ttk.Button(figure_frame, text='T(I)', command=lambda: fig2_1.show())
T_z_button = ttk.Button(figure_frame, text='I(z)', command=lambda: fig2_2.show())
all_button = ttk.Button(figure_frame, text='All runs', command=lambda: fig3.show())
T_I_button.grid(row=1, column=0, padx=5, pady=5)
T_z_button.grid(row=1, column=1, padx=5, pady=5)
all_button.grid(row=1, column=2, padx=5, pady=5)

############################################################
# Export and Run buttons
############################################################
run_button = ttk.Button(window, text='Run', command=lambda: run())
export_button = ttk.Button(window, text='Export', command=lambda: main.export(
    DIRECTORY, TITLE, OUTPUT_STRING, fig1, fig2, fig2_1, fig2_2, fig3, PARAMETER_DATA))
close = ttk.Button(window, text='Close', command=lambda: window.quit())
run_button.grid(row=4, column=1, padx=20, pady=10, sticky='news')
export_button.grid(row=4, column=2, padx=20, pady=10, sticky=tk.W)
close.grid(row=4, column=3, padx=20, pady=10, sticky=tk.E)

window.mainloop()