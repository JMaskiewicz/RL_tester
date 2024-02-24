###
import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import webbrowser
import pandas as pd


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
