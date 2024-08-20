import os
import pandas as pd

def combine_csv_files(root_dir, file_name):
    combined_data = pd.DataFrame()

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file_name in file:
                temp_data = pd.read_csv(os.path.join(subdir, file), encoding='ISO-8859-1')
                combined_data = pd.concat([combined_data, temp_data])

    return combined_data

# Usage
root_dir = '/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/STEaM WP4 team/MSTES-CHP/Win 1D Model/results'
# root_dir = '/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/STEaM WP4 team/MSTES-HP/Energy Flow & MTES/Results'  # replace with your directory path
file_name = 'kpi_last'  # replace with your file name
combined_data = combine_csv_files(root_dir, file_name)
combined_data.to_csv('combined.csv', index=False, encoding='ISO-8859-1')
