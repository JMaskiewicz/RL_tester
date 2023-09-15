# import libraries
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def backtest(df, strategy, starting_capital=10000, provision=0.0001, leverage=10,
                    currencies=['EURUSD', 'USDJPY', 'EURJPY'], **kwargs):
    # Setup DataFrame
    current_index = df.index[0]
    df['Capital'] = 0
    df.loc[current_index, 'Capital'] = starting_capital
    df['Available_Capital'] = 0
    df.loc[current_index, 'Available_Capital'] = starting_capital
    df['Margin'] = 1
    df['Provisions'] = 0

    for currency in currencies:
        df[('Position_proposition', currency)] = 0
        df[('Capital_in', currency)] = 0
        df.loc[current_index, ('Capital_in', currency)] = 0
        df[('PnL', currency)] = 0

    # Backtesting Loop
    for i in tqdm(range(1, len(df))):
        previous_index = current_index
        current_index = df.index[i]

        # Vectorized calculation of required capital
        capital_in_previous = df.loc[previous_index, ('Capital_in', currencies)]
        df.loc[current_index, 'Available_Capital'] = df.loc[previous_index, 'Capital'][0] - abs(
            capital_in_previous).sum()

        # Get new positions from strategy function
        new_positions = strategy(df, current_index, currencies, **kwargs)  # , avaible_capital

        new_positions_series = pd.Series(new_positions).reindex(capital_in_previous.index.get_level_values(1))
        required_capital_series = abs(new_positions_series + capital_in_previous)
        total_required_capital = required_capital_series.sum()

        # Vectorized calculation of provisions
        new_provisions_series = pd.Series(new_positions).reindex(capital_in_previous.index.get_level_values(1))
        provisions_series = abs(new_provisions_series) * provision
        total_provisions = provisions_series.sum()

        # 1. Get the Close prices at the current and previous indexes for all currencies
        current_close_prices = df.loc[current_index, ('Close', currencies)]
        previous_close_prices = df.loc[previous_index, ('Close', currencies)]

        # 2. Get the Capital_in values at the previous index for all currencies
        previous_capital_in = df.loc[previous_index, ('Capital_in', currencies)]

        # 3. Calculate the returns for each currency
        current_returns = ((current_close_prices / previous_close_prices) - 1)

        # 4. Reset the MultiIndex of current_returns to match the MultiIndex of previous_capital_in
        current_returns.index = previous_capital_in.index

        # 5. Calculate the PnL series
        PnL_series = current_returns * previous_capital_in * leverage

        # Sum the PnL for all currencies to get the total PnL
        total_PnL = PnL_series.loc['Capital_in'].sum()

        # Update the dataframe with the calculated PnL for each currency
        df.loc[current_index, ('PnL', currencies)] = PnL_series.loc['Capital_in'].values

        # Update the dataframe with the new positions for each currency
        ordered_positions = [new_positions.get(currency, 0) for currency in currencies]
        df.loc[current_index, ('Position_proposition', currencies)] = ordered_positions

        if total_required_capital <= df.loc[previous_index, 'Capital'][0]:
            df.loc[current_index, 'Provisions'] = total_provisions
            # Vectorized update of Capital_in for each currency
            df.loc[current_index, ('Capital_in', currencies)] = new_positions_series + capital_in_previous
        else:
            # Calculate the net change in positions
            net_change_positions = abs(capital_in_previous + new_positions_series) - abs(capital_in_previous)

            # Flatten the multi-index to a single level
            net_change_positions_flat = net_change_positions.reset_index(level=0, drop=True)

            # Reindex the series to match each other's indices
            net_change_positions_flat = net_change_positions_flat.reindex(new_positions_series.index)

            # Get the positions where net_change_positions > 0
            closing_positions = new_positions_series.where(net_change_positions_flat < 0, 0)

            # for cases where we have 1000 long, and we want to add 2000 short we can only short 1000 if there is no capital avaible
            smaller_abs_values = pd.Series(
                np.minimum(np.abs(closing_positions.values), np.abs(capital_in_previous.values)),
                index=closing_positions.index) * np.sign(new_positions_series)

            # Calculate the total capital from closing positions
            total_closing_capital = abs(smaller_abs_values).sum()

            # Vectorized update of Capital_in for each currency
            df.loc[current_index, ('Capital_in', currencies)] = capital_in_previous + smaller_abs_values

            if total_closing_capital > 0:
                # Calculate the total provisions
                total_provisions = abs(closing_positions).sum() * provision
                df.loc[current_index, 'Provisions'] = total_provisions
            else:
                total_provisions = 0

        df.loc[current_index, 'Capital'] = df.loc[previous_index, 'Capital'][0] - total_provisions + total_PnL

    # Closing all positions in the last observation
    last_index = df.index[-1]

    # 1. Get the Close prices for the last and second last observations for all currencies
    final_close_prices = df.loc[last_index, ('Close', currencies)]
    penultimate_close_prices = df.loc[df.index[-2], ('Close', currencies)]

    # 2. Get the Capital_in values at the second last index for all currencies
    penultimate_capital_in = df.loc[df.index[-2], ('Capital_in', currencies)]

    # 3. Calculate the returns for each currency
    final_returns = ((final_close_prices / penultimate_close_prices) - 1)

    # 4. Reset the MultiIndex of final_returns to match the MultiIndex of penultimate_capital_in
    final_returns.index = penultimate_capital_in.index

    # 5. Calculate the final PnL series
    final_PnL_series = final_returns * penultimate_capital_in * leverage

    # Sum the final PnL for all currencies to get the total final PnL
    total_final_PnL = final_PnL_series.loc['Capital_in'].sum()

    # Update the capital by adding the final PnL
    df.loc[last_index, 'Capital'] = df.loc[df.index[-2], 'Capital'][0] + total_final_PnL

    # Setting the Capital_in values for all currencies to zero for the last observation
    df.loc[last_index, ('Capital_in', currencies)] = 0

    return df