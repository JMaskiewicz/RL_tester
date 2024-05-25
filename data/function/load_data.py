"""
# TODO add description

"""

# import libraries
import pandas as pd
import os
from tqdm import tqdm
import rarfile
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

# Suppress specific warnings from openpyxl
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")

data_folder = "./data"


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


def load_data_2(tickers, timestamp_x):
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    dfs_all = []
    for ticker in tqdm(tickers):
        dfs = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        ticker_folder = os.path.join(project_root, 'data_sets', ticker)

        # Check if ticker folder exists
        if not os.path.exists(ticker_folder):
            print(f"Folder for ticker {ticker} does not exist. Skipping...")
            continue

        # Iterate through each file in the ticker-specific folder
        for file in tqdm(os.listdir(ticker_folder)):
            if file.endswith('.xlsx'):
                file_path = os.path.join(ticker_folder, file)

                # Read the .xlsx file without headers and assign column names
                df = pd.read_excel(file_path, engine='openpyxl', header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                dfs.append(df)

        df_all = pd.concat(dfs)
        df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y-%m-%d %H:%M')
        df_all = df_all.set_index('Date')
        df_all = df_all[['Open', 'High', 'Low', 'Close']]
        df_all = df_all.resample(timestamp_x).agg(agg_dict).dropna()
        df_all['Currency'] = ticker
        df_all.reset_index(inplace=True)
        df_all.set_index(['Date', 'Currency'], inplace=True)
        df_all = df_all.unstack('Currency')
        dfs_all.append(df_all)

    return pd.concat(dfs_all, axis=1)


def process_ticker_xlsx(ticker, timestamp_x, agg_dict, project_root):
    dfs = []
    ticker_folder = os.path.join(project_root, 'data_sets', ticker)

    if not os.path.exists(ticker_folder):
        print(f"Folder for ticker {ticker} does not exist. Skipping...")
        return None

    for file in os.listdir(ticker_folder):
        if file.endswith('.xlsx'):
            file_path = os.path.join(ticker_folder, file)
            df = pd.read_excel(file_path, engine='openpyxl', header=None,
                               names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            dfs.append(df)

    if not dfs:
        return None

    df_all = pd.concat(dfs)
    df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y-%m-%d %H:%M')
    df_all = df_all.set_index('Date')[['Open', 'High', 'Low', 'Close']]
    df_all = df_all.resample(timestamp_x).agg(agg_dict).dropna()
    df_all['Currency'] = ticker
    df_all.reset_index(inplace=True)
    df_all.set_index(['Date', 'Currency'], inplace=True)
    df_all = df_all.unstack('Currency')
    return df_all


def load_data_parallel(tickers, timestamp_x, timestamp_y='M1'):
    start_time = time.time()
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    dfs_all = []
    with ProcessPoolExecutor() as executor:
        # Submit all tasks to the executor
        future_to_ticker = {executor.submit(process_ticker_pkl, ticker, timestamp_x, timestamp_y, agg_dict, project_root): ticker for ticker in tickers}

        # Process as they complete
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc='Processing tickers'):
            df = future.result()
            if df is not None:
                dfs_all.append(df)

    # Concatenate all DataFrames in the list
    if dfs_all:
        df = pd.concat(dfs_all, axis=1)
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Data loaded in {episode_time} seconds")
        return df
    else:
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Data loaded in {episode_time} seconds")
        return pd.DataFrame()

# TODO check parquet
def process_ticker_pkl(ticker, timestamp_x, timestamp_y, agg_dict, project_root):
    dfs = []
    ticker_folder = os.path.join(project_root, 'data_sets', ticker)

    if not os.path.exists(ticker_folder):
        print(f"Folder for ticker {ticker} does not exist. Skipping...")
        return None

    for file in os.listdir(ticker_folder):
        if file.endswith('.pkl'):
            file_path = os.path.join(ticker_folder, file)
            df = pd.read_pickle(file_path)
            dfs.append(df)

    if not dfs:
        return None

    df_all = pd.concat(dfs)
    df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y-%m-%d %H:%M')
    df_all = df_all.set_index('Date')[['Open', 'High', 'Low', 'Close']]
    df_all = df_all.resample(timestamp_x).agg(agg_dict).dropna()
    df_all['Currency'] = ticker
    df_all.reset_index(inplace=True)
    df_all.set_index(['Date', 'Currency'], inplace=True)
    df_all = df_all.unstack('Currency')
    return df_all

if __name__ == '__main__':
    '''    start_time = time.time()
    df = load_data_2(['WTIUSD', 'BCOUSD'], '1H')
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"Data loaded in {episode_time} seconds")'''
    start_time = time.time()
    df_2 = load_data_parallel(['WTIUSD', 'BCOUSD','EURUSD'], '1H')
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"Data loaded in {episode_time} seconds")
