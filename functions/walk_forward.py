# import libraries
import numpy as np
import pandas as pd
from datetime import timedelta

def get_week_indices(df, start_week, num_weeks):
    start_date = df.index[0] + timedelta(weeks=start_week)
    end_date = start_date + timedelta(weeks=num_weeks)
    return start_date, end_date


def split_data_by_weeks(df, train_weeks, test_weeks, val_weeks, forward_shifts):
    """
    Splits the dataframe into training, testing, and validation sets based on the number of weeks.
    The function also shifts forward by a certain number of weeks.
    """
    results = []

    max_date = df.index[-1]  # Get the date of the last entry

    shift_date = df.index[0]
    while True:
        train_start_date = shift_date
        train_end_date = train_start_date + pd.Timedelta(weeks=train_weeks) - pd.Timedelta(days=1)

        test_start_date = train_end_date
        test_end_date = test_start_date + pd.Timedelta(weeks=test_weeks) - pd.Timedelta(days=1)

        val_start_date = test_end_date
        val_end_date = val_start_date + pd.Timedelta(weeks=val_weeks) - pd.Timedelta(days=1)

        train = df[train_start_date: train_end_date]
        test = df[test_start_date: test_end_date]
        val = df[val_start_date: val_end_date]

        results.append((train, test, val))

        # Breaking condition
        if val_end_date >= max_date:
            break

        shift_date += pd.Timedelta(weeks=forward_shifts)

    return results