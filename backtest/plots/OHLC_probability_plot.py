###
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import webbrowser
from threading import Timer

def OHLC_probability_plot(df_train, df_validation, df_test, episode_probabilities, portnumber=8062):
    def get_ohlc_data(selected_dataset, market):
        dataset_mapping = {
            'train': df_train,
            'validation': df_validation,
            'test': df_test
        }
        data = dataset_mapping[selected_dataset]
        ohlc_columns = [('Open', market), ('High', market), ('Low', market), ('Close', market)]
        ohlc_data = data.loc[:, ohlc_columns].reset_index()
        ohlc_data.columns = ['Time'] + [col[0] for col in ohlc_data.columns[1:]]
        ohlc_data['Time'] = pd.to_datetime(ohlc_data['Time'])
        return ohlc_data

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'Train', 'value': 'train'},
                {'label': 'Validation', 'value': 'validation'},
                {'label': 'Test', 'value': 'test'}
            ],
            value='train'
        ),
        dcc.Input(id='episode-input', type='number', value=0, min=0, step=1),
        dcc.Graph(id='probability-plot'),
        dcc.Graph(id='ohlc-plot')
    ])

    @app.callback(
        Output('probability-plot', 'figure'),
        [Input('dataset-dropdown', 'value'), Input('episode-input', 'value')]
    )
    def update_probability_plot(selected_dataset, selected_episode):
        data = episode_probabilities[selected_dataset][selected_episode]
        x_values = list(range(len(data['Short'])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=data['Short'], mode='lines', name='Short'))
        fig.add_trace(go.Scatter(x=x_values, y=data['Neutral'], mode='lines', name='Neutral'))
        fig.add_trace(go.Scatter(x=x_values, y=data['Long'], mode='lines', name='Long'))
        fig.update_layout(title='Action Probabilities Over Episodes', xaxis_title='Time', yaxis_title='Probability', yaxis=dict(range=[0, 1]))
        return fig

    @app.callback(
        Output('ohlc-plot', 'figure'),
        [Input('dataset-dropdown', 'value')]
    )
    def update_ohlc_plot(selected_dataset, market='EURUSD'):
        ohlc_data = get_ohlc_data(selected_dataset, market)
        if ohlc_data.empty:
            return go.Figure(layout=go.Layout(title="No OHLC data available."))
        ohlc_fig = go.Figure(data=[go.Ohlc(
            x=ohlc_data['Time'],
            open=ohlc_data['Open'],
            high=ohlc_data['High'],
            low=ohlc_data['Low'],
            close=ohlc_data['Close']
        )])
        ohlc_fig.update_layout(title='OHLC Data', xaxis_title='Time', yaxis_title='Price')
        return ohlc_fig

    app.run_server(debug=True, port=portnumber)
    webbrowser.open(f"http://127.0.0.1:{portnumber}/")

def PnL_generation_plot(balances_dfs, alternative_strategies=None, port_number=8050):
    records = []
    for (agent_gen, data_set), balances in balances_dfs.items():
        df = pd.DataFrame({'Balance': balances, 'Time Step': range(len(balances))})
        df['Agent Generation'] = agent_gen
        df['DATA_SET'] = data_set
        records.append(df)

    flattened_df = pd.concat(records)

    benchmark_flattened_dfs = []
    if alternative_strategies:
        for strategy in alternative_strategies:
            benchmark_records = []
            for (strategy_name, data_set), balances in strategy.items():
                df = pd.DataFrame({'Benchmark Balance': balances, 'Time Step': range(len(balances))})
                df['Strategy'] = strategy_name
                df['DATA_SET'] = data_set  # Ensure this matches the key used in balances_dfs
                benchmark_records.append(df)
            benchmark_flattened_dfs.append(pd.concat(benchmark_records))
    else:
        benchmark_flattened_dfs = [None]

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H4('Interactive PnL Plot'),
        dcc.Dropdown(
            id='agent-gen-dropdown',
            options=[{'label': f'Agent Generation {i}', 'value': i} for i in flattened_df['Agent Generation'].unique()],
            value=flattened_df['Agent Generation'].unique()[0],
            clearable=False
        ),
        dcc.Dropdown(
            id='data-set-dropdown',
            options=[{'label': ds, 'value': ds} for ds in flattened_df['DATA_SET'].unique()],
            value=flattened_df['DATA_SET'].unique()[0],
            clearable=False
        ),
        dcc.Graph(id='pnl-plot')
    ])

    @app.callback(
        Output('pnl-plot', 'figure'),
        [Input('agent-gen-dropdown', 'value'),
         Input('data-set-dropdown', 'value')]
    )
    def update_graph(selected_agent_gen, selected_data_set):
        filtered_df = flattened_df[
            (flattened_df['Agent Generation'] == selected_agent_gen) & (flattened_df['DATA_SET'] == selected_data_set)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Time Step'], y=filtered_df['Balance'], mode='lines', name='Agent Balance'))

        for benchmark_flattened_df in benchmark_flattened_dfs:
            if benchmark_flattened_df is not None:
                filtered_benchmark_df = benchmark_flattened_df[benchmark_flattened_df['DATA_SET'] == selected_data_set]
                if not filtered_benchmark_df.empty:
                    for strategy_name in filtered_benchmark_df['Strategy'].unique():
                        strategy_df = filtered_benchmark_df[filtered_benchmark_df['Strategy'] == strategy_name]
                        fig.add_trace(go.Scatter(x=strategy_df['Time Step'], y=strategy_df['Benchmark Balance'], mode='lines', name=f'{strategy_name} Balance'))

        fig.update_layout(title='PnL Over Time', xaxis_title='Time Step', yaxis_title='Balance')
        return fig

    def open_browser():
        webbrowser.open_new(f'http://127.0.0.1:{port_number}/')

    Timer(1, open_browser).start()
    app.run_server(debug=True, port=port_number)

def Probability_generation_plot(probs_dfs, port_number=8050):
    # Transforming your data structure into a suitable format for plotting
    records = []
    for (agent_gen, data_set), df in probs_dfs.items():
        df['Agent Generation'] = agent_gen
        df['DATA_SET'] = data_set
        df['Time Step'] = df.index
        records.append(df)

    flattened_df = pd.concat(records)

    # Initialize Dash app
    app = dash.Dash(__name__)

    # App layout
    app.layout = html.Div([
        html.H4('Interactive Probability Plot'),
        dcc.Dropdown(
            id='agent-gen-dropdown',
            options=[{'label': f'Agent Generation {i}', 'value': i} for i in flattened_df['Agent Generation'].unique()],
            value=flattened_df['Agent Generation'].unique()[0],
            clearable=False
        ),
        dcc.Dropdown(
            id='data-set-dropdown',
            options=[{'label': ds, 'value': ds} for ds in flattened_df['DATA_SET'].unique()],
            value=flattened_df['DATA_SET'].unique()[0],
            clearable=False
        ),
        dcc.Graph(id='probability-plot')
    ])

    # Callback to update the graph based on dropdown selections
    @app.callback(
        Output('probability-plot', 'figure'),
        [Input('agent-gen-dropdown', 'value'),
         Input('data-set-dropdown', 'value')]
    )
    def update_graph(selected_agent_gen, selected_data_set):
        filtered_df = flattened_df[
            (flattened_df['Agent Generation'] == selected_agent_gen) & (flattened_df['DATA_SET'] == selected_data_set)]

        fig = go.Figure()
        for col in ['Short', 'Neutral', 'Long']:
            fig.add_trace(go.Scatter(x=filtered_df['Time Step'], y=filtered_df[col], mode='lines', name=col))

        fig.update_layout(title='Probabilities Over Time', xaxis_title='Time Step', yaxis_title='Probability')
        return fig

    # Function to open the web browser and then start the server
    def open_browser():
        webbrowser.open_new(f'http://127.0.0.1:{port_number}/')

    Timer(1, open_browser).start()
    app.run_server(debug=True, port=port_number)


def PnL_generations(backtest_results, port_number=8050):
    app = dash.Dash(__name__)

    # Extract the unique labels
    unique_labels = backtest_results['Label'].unique()

    app.layout = html.Div([
        html.H4('Interactive Balance Comparison Plot'),
        dcc.Dropdown(
            id='label-dropdown',
            options=[{'label': label, 'value': label} for label in unique_labels],
            value=unique_labels[0],  # Set default value to the first label
            clearable=False
        ),
        dcc.Graph(id='balance-comparison-plot')
    ])

    @app.callback(
        Output('balance-comparison-plot', 'figure'),
        [Input('label-dropdown', 'value')]
    )
    def update_graph(selected_label):
        filtered_df = backtest_results[backtest_results[('Label', '')] == selected_label]

        fig = go.Figure()

        for strategy in [strat for strat in backtest_results.columns.get_level_values(0).unique() if
                         strat not in ['', 'Label']]:
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df[(strategy, 'Final Balance')],  # Ensure this matches MultiIndex structure
                mode='lines+markers',
                name=strategy
            ))

        fig.update_layout(
            title=f'Balance Comparison for Label: {selected_label}',
            xaxis_title='Agent Generation',
            yaxis_title='Balance',
            legend_title="Strategy",
            xaxis=dict(type='category')
        )

        return fig

    def open_browser():
        webbrowser.open_new(f'http://127.0.0.1:{port_number}/')

    Timer(1, open_browser).start()
    app.run_server(debug=False, port=port_number)

def Reward_generations(backtest_results, port_number=8051):  # Consider using a different port if running simultaneously
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Extract the unique labels
    unique_labels = backtest_results[('Label', '')].unique()  # Assuming 'Label' extraction is correct

    app.layout = html.Div([
        html.H4('Interactive Reward Comparison Plot'),
        dcc.Dropdown(
            id='label-dropdown-reward',  # Ensure unique ID if running simultaneously with PnL_generations
            options=[{'label': label, 'value': label} for label in unique_labels],
            value=unique_labels[0],  # Default value
            clearable=False
        ),
        dcc.Graph(id='reward-comparison-plot')  # Corrected ID
    ])

    @app.callback(
        Output('reward-comparison-plot', 'figure'),  # Ensure this matches dcc.Graph ID
        [Input('label-dropdown-reward', 'value')]  # Match updated ID
    )
    def update_graph(selected_label):
        filtered_df = backtest_results[backtest_results[('Label', '')] == selected_label]

        fig = go.Figure()

        for strategy in [strat for strat in backtest_results.columns.get_level_values(0).unique() if
                         strat not in ['', 'Label']]:
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df[(strategy, 'Total Reward')],  # Ensure this matches MultiIndex structure
                mode='lines+markers',
                name=strategy
            ))

        fig.update_layout(
            title=f'Reward Comparison for Label: {selected_label}',
            xaxis_title='Agent Generation',
            yaxis_title='Reward',
            legend_title="Strategy",
            xaxis=dict(type='category')
        )

        return fig

    def open_browser():
        webbrowser.open_new(f'http://127.0.0.1:{port_number}/')

    Timer(1, open_browser).start()
    app.run_server(debug=False, port=port_number)


def calculate_drawdowns(balance_series):
    percentage_series = 100 * (balance_series / balance_series.iloc[0] - 1)
    peak = percentage_series.expanding(min_periods=1).max()
    drawdown = percentage_series - peak
    return drawdown

def PnL_drawdown_plot(balances_dfs, alternative_strategies=None, port_number=8050):
    records = []
    for (agent_gen, data_set), balances in balances_dfs.items():
        df = pd.DataFrame({'Balance': balances, 'Time Step': range(len(balances))})
        df['Agent Generation'] = agent_gen
        df['DATA_SET'] = data_set
        df['Drawdown'] = calculate_drawdowns(df['Balance'])
        records.append(df)

    flattened_df = pd.concat(records)

    benchmark_flattened_dfs = []
    if alternative_strategies:
        for strategy in alternative_strategies:
            benchmark_records = []
            for (strategy_name, data_set), balances in strategy.items():
                df = pd.DataFrame({'Benchmark Balance': balances, 'Time Step': range(len(balances))})
                df['Strategy'] = strategy_name
                df['DATA_SET'] = data_set
                df['Drawdown'] = calculate_drawdowns(df['Benchmark Balance'])
                benchmark_records.append(df)
            benchmark_flattened_dfs.append(pd.concat(benchmark_records))
    else:
        benchmark_flattened_dfs = [None]

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H4('Interactive Drawdown Plot'),
        dcc.Dropdown(
            id='agent-gen-dropdown',
            options=[{'label': f'Agent Generation {i}', 'value': i} for i in flattened_df['Agent Generation'].unique()],
            value=flattened_df['Agent Generation'].unique()[0],
            clearable=False
        ),
        dcc.Dropdown(
            id='data-set-dropdown',
            options=[{'label': ds, 'value': ds} for ds in flattened_df['DATA_SET'].unique()],
            value=flattened_df['DATA_SET'].unique()[0],
            clearable=False
        ),
        dcc.Graph(id='drawdown-plot')
    ])

    @app.callback(
        Output('drawdown-plot', 'figure'),
        [Input('agent-gen-dropdown', 'value'),
         Input('data-set-dropdown', 'value')]
    )
    def update_graph(selected_agent_gen, selected_data_set):
        filtered_df = flattened_df[
            (flattened_df['Agent Generation'] == selected_agent_gen) & (flattened_df['DATA_SET'] == selected_data_set)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Time Step'], y=filtered_df['Drawdown'], mode='lines', name='Agent Drawdown'))

        for benchmark_flattened_df in benchmark_flattened_dfs:
            if benchmark_flattened_df is not None:
                filtered_benchmark_df = benchmark_flattened_df[benchmark_flattened_df['DATA_SET'] == selected_data_set]
                if not filtered_benchmark_df.empty:
                    for strategy_name in filtered_benchmark_df['Strategy'].unique():
                        strategy_df = filtered_benchmark_df[filtered_benchmark_df['Strategy'] == strategy_name]
                        fig.add_trace(go.Scatter(x=strategy_df['Time Step'], y=strategy_df['Drawdown'], mode='lines', name=f'{strategy_name} Drawdown'))

        fig.update_layout(title='Drawdown Over Time', xaxis_title='Time Step', yaxis_title='Drawdown (%)')
        return fig

    def open_browser():
        webbrowser.open_new(f'http://127.0.0.1:{port_number}/')

    Timer(1, open_browser).start()
    app.run_server(debug=True, port=port_number)