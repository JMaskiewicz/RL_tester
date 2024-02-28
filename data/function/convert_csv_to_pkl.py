import pandas as pd
import os


# Absolute path to your data_sets directory
data_sets_path = r"C:\\Users\\jmask\\PycharmProjects\\RL_tester\\data\\data_sets"

# Function to convert XLSX to Pickle
def convert_xlsx_to_pkl(asset_dir_path):
    for filename in os.listdir(asset_dir_path):
        if filename.endswith('.xlsx'):
            # Construct full file path
            file_path = os.path.join(asset_dir_path, filename)
            # Read Excel file
            df = pd.read_excel(file_path, engine='openpyxl', header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            # Convert to Pickle, replacing the xlsx extension with pkl
            pickle_path = file_path.replace('.xlsx', '.pkl')
            df.to_pickle(pickle_path)
            print(f"Converted {filename} to Pickle format.")

# Loop through each asset directory in data_sets
for asset_dir in os.listdir(data_sets_path):
    asset_dir_path = os.path.join(data_sets_path, asset_dir)
    if os.path.isdir(asset_dir_path):
        # Convert all XLSX files in this directory to Pickle
        convert_xlsx_to_pkl(asset_dir_path)
        print(f"Processed asset directory: {asset_dir}")