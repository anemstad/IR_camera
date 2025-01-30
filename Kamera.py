# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:08:39 2025

@author: Ane
"""

import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% Read fluke data

fluke_path = r"C:\Users\Ane\OneDrive - Universitetet i Oslo\Documents\Boccara_Master\IR_camera\Data\surface_fluke.csv"

fluke = pd.read_csv(fluke_path)

fluke_time = fluke.iloc[:,0]
fluke_data = fluke.iloc[:,1]

#%% Read camera data
dim =  [46,49]

camera_path = r"C:\Users\Ane\OneDrive - Universitetet i Oslo\Documents\Boccara_Master\IR_camera\Data"
surface_path = camera_path + r"\surface_box.csv"
output_folder = camera_path + r"\Plots"

#%% 

def read_temp_csv(file_path, dimentions):
    
    first_rows_list = []
    data_matrices = []
    
    rows = dimentions[0]
    cols = dimentions[1]
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        next(reader, None) #Skip the first 4 rows with unnecessary data
        next(reader, None)
        next(reader, None)
        next(reader, None) 
        
        while True:
            try:
                # Read the first row and add it to the first_rows_list as a numpy array
                first_row = np.array([str(x) for x in next(reader) if x])  # Convert elements to float
                first_rows_list.append(first_row)
                
                # Read the next 46 rows and store them in a matrix
                data_matrix = []
                for _ in range(rows):
                    row = next(reader)
                    new_row = [float(cell.replace(",", ".")) if cell.replace(",", ".").replace(".", "").isdigit() else cell for cell in row]
                    #row = list(map(float, row))
                    data_matrix.append(new_row)  # Convert elements to float
    
                data_matrices.append(np.array(data_matrix))
                
                # Skip the empty line after each section
                while True:
                    row = next(reader, [])
                    if not row:
                        break               
            except StopIteration:
                break
        
        #Clean data for plotting
        data_matrices = np.array(data_matrices)
        matrix = data_matrices[:len(data_matrices), :dim[0], :dim[1]] #Remove the last empty column
        matrix_plot = matrix.astype(float) #Transform from string to float
        
        return np.array(first_rows_list, dtype=object), matrix_plot

#%% Test read function 

first_rows, full_data_matrix = read_temp_csv(surface_path, dim)

#%%

def plot_temp_matrix(data, output_folder_save):
    
    rocket = sns.color_palette("rocket", as_cmap=True)
    
    i = 0
    while i < len(data):
        
        bounds = [24, 25, 26, 27, 28, 29]
        norm = mpl.colors.BoundaryNorm(bounds, rocket.N, extend='both')
        
        fig, ax = plt.subplots()  # Create figure and axis
        cax = ax.matshow(data[i], cmap=rocket)  # Use colormap
        
        plt.colorbar(mpl.cm.ScalarMappable(norm= norm, cmap = rocket), ax = ax)
        plt.title(str(first_rows[i]))
        plt.savefig(str(output_folder_save + "\\matrix_"+str(i)+".png"))    
        plt.close()
        i +=1
        
#%% Test plot function 

plot_temp_matrix(full_data_matrix, output_folder)

