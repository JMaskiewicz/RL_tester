import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os

def fetch_crypto_data(api_key, fsym, tsym, from_time, days_span=1500):
    to_time = from_time + timedelta(days=days_span).total_seconds()
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    data = []
    while from_time < to_time:
        params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': 20,  # max limit
            'toTs': to_time,
            'api_key': api_key
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}, {response.text}")
            break

        response_json = response.json()
        if 'Data' not in response_json or 'Data' not in response_json['Data']:
            print("Unexpected API response format:", response_json)
            break

        batch = response_json['Data']['Data']
        data.extend(batch)
        to_time = batch[-1]['time'] - 1  # adjust to_time for the next batch, if needed

    return pd.DataFrame(data)

def save_yearly_data(data, asset_name):
    if data.empty:
        print(f"No data available for {asset_name}.")
        return

    data.drop_duplicates(subset='time', keep='first', inplace=True)  # Remove duplicates
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    base_dir = os.path.join('data', 'data_sets', f'{asset_name}USD')

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for year in data.index.year.unique():
        yearly_data = data[data.index.year == year]
        file_name = f'DAT_XLSX_{asset_name}_M1_{year}.xlsx'
        full_path = os.path.join(base_dir, file_name)
        yearly_data.to_excel(full_path, header=False, index=False)

# Example usage
api_key = '2b1a99ecceac624ae9b41503e1000d4cea752f1141cd9f7db108c833a091dccd'
fsym_list = ['BTC', 'ETH']
tsym = 'USD'
start_time = datetime(2019, 1, 1)
end_time = datetime.now()

for fsym in fsym_list:
    current_time = start_time
    all_data = pd.DataFrame()
    while current_time < end_time:
        print(f"Fetching data for {fsym} from {current_time}")
        from_time = int(current_time.timestamp())
        fetched_data = fetch_crypto_data(api_key, fsym, tsym, from_time)
        all_data = pd.concat([all_data, fetched_data])
        current_time += timedelta(days=1500)  # increment by 1500 days

    all_data = all_data.drop_duplicates(subset='time', keep='first')  # Remove any duplicates
    #save_yearly_data(all_data, fsym)