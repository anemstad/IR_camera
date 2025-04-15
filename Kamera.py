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
import statistics
import random
import scipy.stats 
import datetime
import numpy as np
import os
from os import listdir
from itertools import islice

def get_filenames(string):
    """
    Fetch all file names of csv files in current directory containing choosen "string".

    Return a list of file names. Might not be sorted!
    """
    files = os.listdir()
    files = [name for name in files if ".csv" in name]
    files = [name for name in files if string in name]
    
    return files

def read_temp_csv(file_path):
    """
    Reads a Hikmicro temperature matrix CSV file, determines frame size,
    and returns time data and a cleaned temperature matrix.
    """
    frame_size = (0, 0)
    first_rows_list = []
    data_matrices = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Determine frame size with minimal processing
    frames = []
    current_frame = []
    
    for line in lines:
        line = line.strip()
        if line and "," in line and any(char.isdigit() for char in line):
            current_frame.append(line)
        elif current_frame:
            frames.append(current_frame)
            current_frame = []
    
    if current_frame:
        frames.append(current_frame)
    
    if frames:
        frame_size = (len(frames[0]), (max(len(row.split(",")) for row in frames[0])) - 2)
    
    #The frame_size variable contain one extra column
    
    rows = frame_size[0]
    
    #Reads the actual data from the file
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        for _ in range(4):  # Skip unnecessary metadata rows
            next(reader, None)
        
        try:
            while True:
                first_row = next(reader, None)
                if first_row is None:
                    break
                first_rows_list.append(first_row[0][16:28])  # Extract time data
                
                data_matrix = []
                for _ in range(rows):
                    row = next(reader, [])
                    new_row = []
                    for cell in row:
                        cell = cell.replace(",", ".").strip()
                        if cell.replace(".", "").isdigit():
                            new_row.append(float(cell))
                        else:
                            new_row.append(np.nan)  # Use NaN for empty/non-numeric values
                    data_matrix.append(new_row)
                
                data_matrices.append(np.array(data_matrix, dtype=float))
                
                # Skip empty lines efficiently
                while next(reader, None):
                    pass
        except StopIteration:
            pass
        
        
        
        matrix = np.array(data_matrices)
        matrix = matrix[:,:,:-1]
        
        
    return first_rows_list, matrix #removes the extra coloumn with na values)

def bin_frames_and_extract_temperatures(data, frames_per_bin, threshold):
    total_frames, height, width = data.shape
    number_of_bins = total_frames // frames_per_bin
    reshaped_data = data[:number_of_bins * frames_per_bin].reshape(number_of_bins, frames_per_bin, height, width)
    mean_temperatures = np.mean(reshaped_data, axis=1)

    temperatures_over_threshold = []

    for i in range(number_of_bins):
        temps_over_threshold = mean_temperatures[i][mean_temperatures[i] > threshold]
        temperatures_over_threshold.append(temps_over_threshold)

    return temperatures_over_threshold, mean_temperatures

def plot_temperature_boxplots(temperatures_over_threshold, mean_temperatures, time_values):
    '''
    List of time values given as strings in HH:MM:SS.sss format
    '''
    
    # Convert time strings to datetime objects
    times = pd.to_datetime(time_values, format='%H:%M:%S.%f')

    # Calculate duration in minutes
    duration_seconds = (times[-1] - times[0]).total_seconds()
    duration_minutes = duration_seconds / 60
    
    # Calculate the duration for each bin in minutes
    bins_count = len(temperatures_over_threshold)
    bin_duration_minutes = duration_minutes / bins_count
    
    # Generate the time intervals as floats for plotting
    time_intervals_minutes = np.arange(bins_count) * bin_duration_minutes

    # Calculate delta temperature change (from first to last mean temperature)
    first_temp = np.mean(mean_temperatures[0])
    last_temp = np.mean(mean_temperatures[-1])
    delta_temp = last_temp - first_temp
    
    # Plot the boxplots with scaled time intervals in minutes
    plt.figure(figsize=(12, 6))
    plt.boxplot(temperatures_over_threshold, positions=time_intervals_minutes, widths=bin_duration_minutes / 2)
    plt.xticks(ticks=time_intervals_minutes, labels=[f'{t:.2f} min' for t in time_intervals_minutes], rotation=45)
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Values Above Threshold Over Time\nDelta Temperature: {delta_temp:.2f} °C')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_multiple_temperature_boxplots(temperature_data_list, time_values, frames_per_bin, threshold):
    times = pd.to_datetime(time_values, format='%H:%M:%S.%f')
    duration_seconds = (times[-1] - times[0]).total_seconds()
    duration_minutes = duration_seconds / 60
    
    bins_count = min(data.shape[0] // frames_per_bin for data in temperature_data_list)
    bin_duration_minutes = duration_minutes / bins_count
    time_intervals_minutes = np.arange(bins_count) * bin_duration_minutes

    plt.figure(figsize=(12, 6))

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i, temperature_data in enumerate(temperature_data_list):
        temperatures_over_threshold, mean_temperatures = bin_frames_and_extract_temperatures(temperature_data, frames_per_bin, threshold)
        
        # Calculate delta temperature change for each dataset
        first_temp = np.mean(mean_temperatures[0])
        last_temp = np.mean(mean_temperatures[-1])
        delta_temp = last_temp - first_temp

        # Plot each dataset's boxplots
        plt.boxplot(temperatures_over_threshold, positions=time_intervals_minutes,
                    widths=bin_duration_minutes / 2, patch_artist=True,
                    boxprops=dict(facecolor=colors[i % len(colors)]),
                    label=f'Dataset {i+1} - Delta Temp: {delta_temp:.2f} °C')

    plt.xticks(ticks=time_intervals_minutes, labels=[f'{t:.2f} min' for t in time_intervals_minutes], rotation=45)
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Values Above Threshold Over Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_multiple_temperature_boxplots(filenames, frames_per_bin, threshold):
    temperature_data_list = []
    all_time_values = []

    for filename in filenames:
        time_values, temperature_data = read_temp_csv(filename)
        temperature_data_list.append(temperature_data)
        all_time_values.append(time_values)

    # Assume the first file for duration calculation and time values usage
    times = pd.to_datetime(all_time_values[0], format='%H:%M:%S.%f')
    duration_seconds = (times[-1] - times[0]).total_seconds()
    duration_minutes = duration_seconds / 60
    
    bins_count = min(data.shape[0] // frames_per_bin for data in temperature_data_list)
    bin_duration_minutes = duration_minutes / bins_count
    time_intervals_minutes = np.arange(bins_count) * bin_duration_minutes

    plt.figure(figsize=(12, 6))

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i, temperature_data in enumerate(temperature_data_list):
        temperatures_over_threshold, mean_temperatures = bin_frames_and_extract_temperatures(temperature_data, frames_per_bin, threshold)
        
        # Calculate delta temperature change for each dataset
        first_temp = np.mean(mean_temperatures[0])
        last_temp = np.mean(mean_temperatures[-1])
        delta_temp = last_temp - first_temp

        # Plot each dataset's boxplots
        plt.boxplot(temperatures_over_threshold, positions=time_intervals_minutes,
                    widths=bin_duration_minutes / 2, patch_artist=True,
                    boxprops=dict(facecolor=colors[i % len(colors)]),
                    label=f'Dataset {i+1} - Delta Temp: {delta_temp:.2f} °C')

    plt.xticks(ticks=time_intervals_minutes, labels=[f'{t:.2f} min' for t in time_intervals_minutes], rotation=45)
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Values Above Threshold Over Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

#### 

def bin_frames_and_compute_mean(data, frames_per_bin):
    total_frames, height, width = data.shape
    
    # Calculate the number of complete bins we can form
    number_of_bins = total_frames // frames_per_bin
    
    # Reshape the data to shape (number_of_bins, frames_per_bin, height, width)
    reshaped_data = data[:number_of_bins * frames_per_bin].reshape(number_of_bins, frames_per_bin, height, width)
    
    # Compute the mean across the specified axis (the frame axis after reshaping)
    mean_temperatures = np.mean(reshaped_data, axis=1)
    
    return mean_temperatures

def get_max_values(matrix):
    """
    Inputs a 3-dimentional hikmicro temperature matrix
    Returns a list of the maximum temperature in each second.
    """
    max_values = []
    for frames in matrix:
        liste = []
        for rows in frames:
            liste.append(max(rows))
        max_values.append(max(liste))
    
    return max_values

def get_pup_values(matrix, cutoff):
    
    
    pup_values = []
    
    i = 0
    for frames in matrix:
        liste = []
        for rows in frames:
            for values in rows:
                if values > cutoff:
                    liste.append(values)
        pup_values.append(liste)
        i += 1
    return pup_values
    

def format_time (time_vector):
    """
    Inputs the time vector extracted from the raw hikmicro csv,
    and formattes it until a vector that can be plotted.
    Returns: Plottable vector, total time variable and the framerate
    """
    
    hours = []
    minutes = []
    seconds = []
    millis = []
    
    for times in time_vector:
        samples = len(time_vector) - 1
        
        hours.append(int(times[:2]))
        minutes.append(int(times[3:5]))
        seconds.append(int(times[6:8]))
        millis.append(int(times[9:12]))
        
    # In milliseconds
    d_hours = (hours[samples] - hours[0]) *(60*1000)
    d_minutes = (minutes[samples] - minutes[0]) *(3600*1000)
    d_seconds = (seconds[samples] - seconds[0]) * 1000
    d_millis = millis[samples] - millis[0]
    
    #Calculate total time in minutes
    total_time = float((d_hours + d_minutes + d_seconds + d_millis) / (1000*3600))
    
    #Calulate rate in samples per seconds
    rate = round((total_time * 60 )/ len(time_vector),2)
    
    
    # Convert time strings to datetime objects
    fmt = "%H:%M:%S.%f"
    time1_plot = [datetime.datetime.strptime(t, fmt) for t in time_vector]
    
    time_vector_plot = pd.Series(time1_plot)
    
    return time_vector_plot, total_time, rate #In minutes float


# Function to plot the matrix, so that each frame have its own plot

def plot_temp_matrix(data, output_folder_save, time_values):
    
    rocket = sns.color_palette("rocket", as_cmap=True)
    
    i = 0
    while i < len(data):
        
        bounds = [24, 25, 26, 27, 28, 29]
        norm = mpl.colors.BoundaryNorm(bounds, rocket.N, extend='both')
        
        fig, ax = plt.subplots()  # Create figure and axis
        cax = ax.matshow(data[i], cmap=rocket)  # Use colormap
        
        plt.colorbar(mpl.cm.ScalarMappable(norm= norm, cmap = rocket), ax = ax)
        plt.title(str(time_values[i]))
        plt.savefig(str(output_folder_save + "\\matrix_"+str(i)+".png"))    
        plt.close()
        i +=1


# Extract random coordinates from the entire frame to perform a test 

def get_test(data, dim, number_of_test, background_temp):
    
    test_values = []
    
    i = 0 
    while i < number_of_test:
    
        r = random.randint(0,(dim[0]-1))
        c = random.randint(0, (dim[1]-1))
        
        value = data[:,r,c]
    
        if value[0] > background_temp:
            test_values.append(value)
        else: i -= 1
        
        i += 1
        
    return (test_values)

# Get basic statictics and place in dataframe to inspect

def get_statistics(test_values):
    
    df = pd.DataFrame(data=None, index = range(0,len(test_values),1), columns=["mean", "median", "std", "variance"])
    
    i = 0
    for t in test_values:
        df.iloc[i,0] = statistics.mean(t)
        df.iloc[i,1] = statistics.mean(t)
        df.iloc[i,2] = statistics.stdev(t)
        df.iloc[i,3] = statistics.variance(t)
        i += 1
        
    return df


# Get all values from the entire matrix for testing

def get_all_test(data, dim):
    
    test_values = []
    x = 0 
    
    while x < dim[0]:
        y = 0 
        
        while y < dim[1]:
            value = data[:,x,y]
            test_values.append(value)
            
            y += 1
            
        x += 1
        
    return (test_values)



#Extract a subset of the matrix

def get_selection(data, dim, rows, colums):
    
    test_values = []
    x = 0 
    
    data = data[:, rows[0]:rows[1], colums[0]:colums[1]]
    
    while x < dim[0]:
        y = 0 
        
        while y < dim[1]:
            value = data[:,x,y]
            test_values.append(value)
            
            y += 1
            
        x += 1
        
    return (test_values)

#Get statistics from a matrix subset of the main matrix
def get_statistics_flat(test_values):
    
    df = pd.DataFrame(data=None, index = range(0,len(test_values),1), columns=["mean", "median", "std", "variance"])
    
    i = 0
    for t in test_values:
        
        flat = t.flatten()
        
        df.iloc[i,0] = statistics.mean(flat)
        df.iloc[i,1] = statistics.median(flat)
        df.iloc[i,2] = statistics.stdev(flat)
        df.iloc[i,3] = statistics.variance(flat)
        i += 1
        
    return df


# Significance test (t-test)

#Inputs both recorded data and referance data
def significance(data, fluke_data):
    p_values = []
    good = 0
    bad = 0

    alpha = 0.05

    i = 0
    
    for t in data:
        while i < len(data):
            
            p = scipy.stats.ttest_ind(t, fluke_data).pvalue
            p = p.item() #kansje gjøres mer?
            p_values.append(p)
            
            if float(p) <= alpha:
                bad += 1
            else: good += 1
            
            i += 1
        
    return p_values, [good, bad]
