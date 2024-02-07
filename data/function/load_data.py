# description
"""

"""

# import libraries
import pandas as pd
import os
from tqdm import tqdm
import rarfile

data_folder = "./data"


def load_data(currencies, timestamp_x):
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    dfs = []
    for currency in tqdm(currencies):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_folder = os.path.join(project_root, 'data_sets')
        file_path = os.path.join(data_folder, f'{currency}_2003-2022.txt')
        df = pd.read_csv(file_path, sep=',')
        df['Date'] = df['Date'] + " " + df['time']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        df = df.drop(["time"], axis=1)
        df.set_index('Date', inplace=True)
        # Convert the naive datetime index to UTC and then to Warsaw time
        # df.index = df.index.tz_localize('America/New_York').tz_convert('Europe/Warsaw')
        df = df[['Open', 'High', 'Low', 'Close']]
        df = df.resample(timestamp_x).agg(agg_dict)
        df = df.dropna()
        df['Currency'] = currency
        df.reset_index(inplace=True)
        df.set_index(['Date', 'Currency'], inplace=True)
        df = df.unstack('Currency')
        dfs.append(df)
    df_all = pd.concat(dfs, axis=1)
    return df_all


def unpack_all_rars_in_folder():
    # Use os.path.abspath to get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data_sets folder relative to the script's location
    data_folder = os.path.join(script_dir, '..', 'data_sets')

    print(f"Data folder: {data_folder}")

    # Check if the data folder exists
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return

    # Iterate over all files in the data folder
    for filename in tqdm(os.listdir(data_folder)):
        if filename.endswith('.rar'):
            rar_file_path = os.path.join(data_folder, filename)
            currency_name = os.path.splitext(filename)[0]  # Assuming the rar file name is the currency name
            extracted_folder = os.path.join(data_folder, currency_name)

            print(f"Processing {rar_file_path}")

            # Ensure the extraction directory exists
            if not os.path.exists(extracted_folder):
                os.makedirs(extracted_folder)

            # Extract rar file
            try:
                with rarfile.RarFile(rar_file_path, 'r') as rar_ref:
                    rar_ref.extractall(extracted_folder)
                print(f"Extracted {filename} to {extracted_folder}")
            except Exception as e:
                print(f"Error extracting {filename}: {e}")

data_folder = "./data"

def load_data_long_format(currencies, timestamp_x):
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    dfs = []
    for currency in tqdm(currencies):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_folder = os.path.join(project_root, 'data_sets')
        file_path = os.path.join(data_folder, f'{currency}_2003-2022.txt')

        df = pd.read_csv(file_path, sep=',')
        df['Date'] = pd.to_datetime(df['Date'] + " " + df['time'], format='%Y.%m.%d %H:%M')
        df.drop(["time"], axis=1, inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close']]
        df.set_index('Date', inplace=True)

        # Resample and aggregate
        df = df.resample(timestamp_x).agg(agg_dict).dropna()

        # Add currency identifier
        df['Currency'] = currency

        dfs.append(df.reset_index())  # Reset index to turn 'Date' back into a column

    # Concatenate all dataframes
    df_all = pd.concat(dfs, axis=0)

    return df_all