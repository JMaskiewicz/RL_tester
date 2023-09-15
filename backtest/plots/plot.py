import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from technical_analysys.add_indicators import add_indicators, compute_volatility


def add_trace_based_on_data(fig, df, currency, currencies, row, col):
    if all(col in df.columns for col in
           [('Open', currency), ('High', currency), ('Low', currency), ('Close', currency)]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open', currency],
                high=df['High', currency],
                low=df['Low', currency],
                close=df['Close', currency],
                name=f'Candlestick {currency}',
                visible=(currency == currencies[0])
            ), row=row, col=col)
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close', currency],
                mode='lines',
                name=f'Close {currency}',
                visible=(currency == currencies[0])
            ),
            row=row, col=col
        )


def plot_financial_data(df, strategy, currencies, volatility='garman_klass_volatility', n=200):
    pio.renderers.default = "browser"

    for currency in currencies:
        df['Volatility', currency] = compute_volatility(df, currency, method_func=volatility, n=n)

    for currency in currencies:
        df['Volume', currency] = 0

    for currency in currencies:
        df['Cumulative PnL', currency] = df['PnL', currency].cumsum()

    df['Global PnL', ''] = sum([df['PnL', currency] for currency in currencies]).cumsum()

    all_currencies = currencies + ['Global PnL']

    fig = make_subplots(
        rows=7,
        cols=1,
        vertical_spacing=0.05,
        shared_xaxes=True,
        subplot_titles=['' for _ in range(7)],
        row_heights=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    )

    for currency in currencies:
        add_trace_based_on_data(fig, df, currency, currencies, 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volatility', currency], mode='lines', name=f'Volatility {currency}', visible=(currency == currencies[0])), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume', currency], mode='lines', name=f'Volume {currency}', visible=(currency == currencies[0])), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative PnL', currency], mode='lines', name=f'PnL {currency}', visible=(currency == currencies[0])), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Capital_in', currency], mode='lines', name=f'Position Size {currency}', visible=(currency == currencies[0])), row=6, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Global PnL', ''], mode='lines', name='Global PnL', visible=True), row=7, col=1)

    for i, trace in enumerate(fig.data):
        print(f"Trace {i}:")
        print(trace)
        print("------")

    buttons_currency = []

    for i, currency in enumerate(currencies):
        visibility_list = [currency == all_currencies[j] for j in range(len(currencies)) for _ in range(5)]
        visibility_list.append(True)
        button = {
            "label": currency,
            "method": "update",
            "args": [
                {"visible": visibility_list},
                {"title": f"Currency: {currency}"}
            ]
        }
        buttons_currency.append(button)

    fig.update_layout(
        margin=dict(l=20, r=100, t=20, b=20),
        updatemenus=[{
            "buttons": buttons_currency,
            "direction": "down",
            "showactive": True,
            "x": 1.02,
            "xanchor": "left",
            "y": 0.9,
            "yanchor": "top"
        }],
        annotations=[
            dict(text="Close", x=0, y=0.97, showarrow=False, xref="paper", yref="paper", xanchor="left",
                 yanchor="middle", font=dict(size=14), xshift=-40, textangle=-90),
            dict(text="Volatility", x=0, y=0.69, showarrow=False, xref="paper", yref="paper", xanchor="left",
                 yanchor="middle", font=dict(size=14), xshift=-40, textangle=-90),
            dict(text="Volume", x=0, y=0.52, showarrow=False, xref="paper", yref="paper", xanchor="left",
                 yanchor="middle", font=dict(size=14), xshift=-40, textangle=-90),
            dict(text="PnL", x=0, y=0.35, showarrow=False, xref="paper", yref="paper", xanchor="left", yanchor="middle",
                 font=dict(size=14), xshift=-40, textangle=-90),
            dict(text="Position Size", x=0, y=0.18, showarrow=False, xref="paper", yref="paper", xanchor="left",
                 yanchor="middle", font=dict(size=14), xshift=-40, textangle=-90),
            dict(text="Global PnL", x=0, y=0.01, showarrow=False, xref="paper", yref="paper", xanchor="left",
                 yanchor="middle", font=dict(size=14), xshift=-40, textangle=-90)
        ]
    )

    fig.show()
