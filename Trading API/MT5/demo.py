import MetaTrader5 as mt5
# Your DDQN agent code here

# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Connect to the trading account
account = 123456  # Your account
authorized = mt5.login(account, password="your_password")
if authorized:
    print("Connected to account #", account)
else:
    print("Failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))

# Assuming your agent decides to buy
action = agent.choose_action(current_state)
symbol = "EURUSD"
lot = 0.1  # The volume of the trade

if action == 2:  # Assuming 2 is the 'Buy' action
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(symbol).ask,
        "sl": 0,  # Stop loss
        "tp": 0,  # Take profit
        "deviation": 20,
        "magic": 234000,
        "comment": "DDQN trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send the trade request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Trade failed, retcode={}".format(result.retcode))
    else:
        print("Trade successful")

# Close the MT5 connection
mt5.shutdown()