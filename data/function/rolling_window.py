import re
import pandas as pd


def parse_time_offset(time_value):
    match = re.match(r"(\d+)([YMDH])", time_value)
    if not match:
        raise ValueError(f"Invalid time format: {time_value}")

    number, unit = int(match.group(1)), match.group(2)
    if unit == 'Y':
        return pd.DateOffset(years=number)
    elif unit == 'M':
        return pd.DateOffset(months=number)
    elif unit == 'D':
        return pd.DateOffset(days=number)
    elif unit == 'H':
        return pd.DateOffset(hours=number)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

def rolling_window_datasets(df, window_size='3M', look_back=0):
    start_date = df.index.min()
    end_date = df.index.max()
    rolling_windows = []

    time_offset = parse_time_offset(window_size)

    current_start_date = start_date
    while current_start_date < end_date:
        current_end_date = min(current_start_date + time_offset, end_date)

        window_df = df.loc[current_start_date:current_end_date]

        if look_back > 0:
            first_obs_index = df.index.get_loc(window_df.index[0])
            look_back_start_index = max(first_obs_index - look_back, 0)
            look_back_df = df.iloc[look_back_start_index:first_obs_index]
            window_df = pd.concat([look_back_df, window_df])

        rolling_windows.append(window_df)
        current_start_date = current_end_date

    return rolling_windows