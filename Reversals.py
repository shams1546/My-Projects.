import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from scipy.stats import linregress
from tqdm import tqdm
from untrade.client import Client
import warnings
warnings.filterwarnings('ignore')

def calculate_fibonacci_levels(high, low):
    fib_levels = [0.382, 0.5, 0.618]
    return [low + (high - low) * level for level in fib_levels]

def calculate_obv(df):
    """Compute Standard OBV."""
    obv = [0]  # Initialize OBV with zero
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:  # Price Up
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:  # Price Down
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:  # No Change
            obv.append(obv[-1])
    df["OBV"] = obv
    return df

def calculate_vwobv(df, lookback_period=10):
    """Compute Volume-Weighted OBV (VWOBV) to normalize against total volume."""
    df = calculate_obv(df)  # Compute standard OBV
    df["VWOBV"] = df["OBV"] / df["volume"].rolling(window=lookback_period).sum()
    return df

def calculate_trend(df, threshold=0.009, window=10, lookback_period=10):
    """Compute VWOBV & Trend Classification using OBV Slope."""
    
    def vwobv_slope(series, window=window):
        if len(series) < window:
            return np.nan  # Not enough data
        y = series[-window:]  # Last 'window' VWOBV values
        x = np.arange(window)  # Time steps
        slope, _, _, _, _ = linregress(x, y)  # Linear regression
        return slope
    
    df = calculate_vwobv(df, lookback_period)  # Use VWOBV instead of OBV
    df["VWOBV_Slope"] = df["VWOBV"].rolling(window=window).apply(vwobv_slope, raw=True)

    # Classify Market Trend based on VWOBV slope
    df['Trend'] = 0  # Default: Consolidation
    df.loc[df['VWOBV_Slope'] > threshold, 'Trend'] = 1  # Uptrend
    df.loc[df['VWOBV_Slope'] < -threshold, 'Trend'] = -1  # Downtrend
    return df

def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['close'], candle['open'])
    lower_wick = min(candle['close'], candle['open']) - candle['low']

    if lower_wick >= 2 * body and upper_wick <= body:
        return "Hammer"
    elif upper_wick >= 2 * body and lower_wick <= body:
        return "Inverted Hammer"
    return None

def classify_hammer(candle):
    """
    Classifies Hammer as Strong or Mediocre based on Fibonacci retracement levels.
    """
    if candle['Hammer']:
        fib_382 = candle['low'] + 0.382 * (candle['high'] - candle['low'])
        fib_50 = candle['low'] + 0.5 * (candle['high'] - candle['low'])

        body_bottom = min(candle['open'], candle['close'])

        if body_bottom > fib_382:
            return "Strong Hammer"
        elif body_bottom > fib_50:
            return "Mediocre Hammer"

    elif candle['Inverted_Hammer']:
        fib_382 = candle['high'] - 0.382 * (candle['high'] - candle['low'])
        fib_50 = candle['high'] - 0.5 * (candle['high'] - candle['low'])

        body_top = max(candle['open'], candle['close'])

        if body_top < fib_382:
            return "Strong Inverted Hammer"
        elif body_top < fib_50:
            return "Mediocre Inverted Hammer"
    
    return "None"

def is_strong_bullish(row, volume_ma=None, body_threshold=0.55, wick_threshold=0.25, volume_multiplier=1.2):
    body = row['close'] - row['open']
    upper_wick = row['high'] - row['close']
    lower_wick = row['open'] - row['low']
    body_size = row['high'] - row['low']
    
    # Avoid division errors for tiny candles
    if body_size == 0:
        return False
    
    return (
        body > body_threshold * body_size and  
        upper_wick < wick_threshold * body_size and  
        lower_wick < wick_threshold * body_size and  
        (row['volume'] > volume_multiplier * volume_ma if volume_ma else True)
    )

def is_strong_bearish(row, volume_ma=None, body_threshold=0.55, wick_threshold=0.25, volume_multiplier=1.2):
    body = row['open'] - row['close']
    upper_wick = row['high'] - row['open']
    lower_wick = row['close'] - row['low']
    body_size = row['high'] - row['low']
    
    # Avoid division errors for tiny candles
    if body_size == 0:
        return False
    
    return (
        body > body_threshold * body_size and  
        upper_wick < wick_threshold * body_size and  
        lower_wick < wick_threshold * body_size and  
        (row['volume'] > volume_multiplier * volume_ma if volume_ma else True)
    )

def calculate_stop_loss(high, low, entry_price, trade_type, fib_level=0.382):
    # Calculate Fibonacci retracement levels
    fib_retracement = high - (high - low) * fib_level

    if trade_type == 1:
        sl = min(entry_price, fib_retracement)  # SL below entry
    elif trade_type == 2:
        sl = max(entry_price, fib_retracement)  # SL above entry
    else:
        raise ValueError("Invalid trade type! Use 'buy' or 'sell'.")

    return  round (sl, 4)  # Round for better readability

def identify_early_pullbacks(df,  volume_ma_period=10, volume_threshold=0.5):
    # Compute Moving Average of Volume
    df['Volume_MA'] = ta.SMA(df['volume'], volume_ma_period)

    # Identify Low-Volume Pullbacks (Volume below threshold)
    df['Low_Volume_Pullback'] = df['volume'] < (df['Volume_MA'] * volume_threshold)
    df['pullback_uptrend'] =  df['Low_Volume_Pullback']
    df['pullback_downtrend'] =  df['Low_Volume_Pullback']
    return df

# Smart Trailing Stop Implementation with Early Pullback Detection
def apply_smart_trailing_stop(df):
    long_stop_levels = []
    short_stop_levels = []
    active_long_stop = None
    active_short_stop = None
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        atr = df['ATR'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]
        open = df['open'].iloc[i]
        atr_mult = df['adaptive_multiplier'].iloc[i]
        
        signal = df['signals'].iloc[i]

        # Reset active stops on new trade
        if signal == 1:  # Long entry
            active_long_stop = None
            active_short_stop = None  # reset short stop
        elif signal == 2:  # Short entry
            active_short_stop = None
            active_long_stop = None  # reset long stop
        
        # Predict pullback for long positions
        if df['pullback_uptrend'].iloc[i]:
            new_long_stop = min( close, open) - (atr_mult * atr)
            if active_long_stop is None or new_long_stop > active_long_stop:
                active_long_stop = new_long_stop
        
        # Predict pullback for short positions
        if df['pullback_downtrend'].iloc[i]:
            new_short_stop = max( close, open)  + (atr_mult * atr)
            if active_short_stop is None or new_short_stop < active_short_stop:
                active_short_stop = new_short_stop
        
        long_stop_levels.append(active_long_stop if active_long_stop is not None else np.nan)
        short_stop_levels.append(active_short_stop if active_short_stop is not None else np.nan)
    
    df['smart_trailing_stop_long'] = long_stop_levels
    df['smart_trailing_stop_short'] = short_stop_levels
    
    df['smart_trailing_stop_long'] = df['smart_trailing_stop_long'].ffill()
    df['smart_trailing_stop_short'] = df['smart_trailing_stop_short'].ffill() 
    
    return df

# PROCESS_DATA
def process_data(df):
    
    df = df.rename(columns={'datetime': 'timestamp'})

    # Heikin-Ashi Calculation
    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_Open'] = (df['open'] + df['close']) / 2
    df['HA_Open'] = df['HA_Open'].shift(1).fillna(df['open'])
    df['HA_High'] = df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close', 'low']].min(axis=1)

    #Technical Indicators
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # ADX, +DI, and -DI Calculation
    df['ADX'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['+DI'] = ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['-DI'] = ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    
  # **Reversal Signals**
    df['Go_Uptrend'] = (df['-DI'].shift(1) > df['+DI'].shift(1)) & (df['-DI'] < df['+DI'])
    df['Go_Downtrend'] = (df['+DI'].shift(1) > df['-DI'].shift(1)) & (df['+DI'] < df['-DI'])

    df = calculate_trend(df)
    
    # Apply Hammer Detection
    df['Hammer'] = df.apply(lambda row: is_hammer(row) == "Hammer", axis=1)
    df['Inverted_Hammer'] = df.apply(lambda row: is_hammer(row) == "Inverted Hammer", axis=1)

    df['Strong_Hammer'] = df.apply(lambda row: classify_hammer(row) == "Strong Hammer", axis=1)
    df['Mediocre_Hammer'] = df.apply(lambda row: classify_hammer(row) == "Mediocre Hammer", axis=1)
    df['Mediocre_Inverted_Hammer'] = df.apply(lambda row: classify_hammer(row) == "Mediocre Inverted Hammer", axis=1)
    df['Strong_Inverted_Hammer'] = df.apply(lambda row: classify_hammer(row) == "Strong Inverted Hammer", axis=1)

    df['Strong_Bullish_Next'] = df['Mediocre_Hammer'].shift(1) & df.apply(lambda row: is_strong_bullish(row), axis=1)
    df['Strong_Bearish_Next'] = df['Mediocre_Inverted_Hammer'].shift(1) & df.apply(lambda row: is_strong_bearish(row), axis=1)
    
    # Calculate relative volume
    df['avg_volume'] = df['volume'].rolling(window=10).mean()
    df['relative_volume'] = df['volume'] / df['avg_volume']

    # Adaptive multiplier based on relative volume
    df['adaptive_multiplier'] = np.where(df['relative_volume'] > 1.5, 1.0,    # High volume
                                     np.where(df['relative_volume'] < 0.7, 0.6,  # Low volume
                                              0.8))  # Medium volume 

    df['No_Demand'] = (df['high'] - df['low'] < df['ATR'] * df['adaptive_multiplier']) & \
                  (df['volume'] < df['volume'].rolling(3).mean()) & \
                  ((df['close'] - df['low']) / (df['high'] - df['low']) > df['close'].rolling(3).mean() / df['high'].rolling(3).mean()) 
    df['No_Supply'] = (df['high'] - df['low'] < df['ATR'] * df['adaptive_multiplier']) & \
                  (df['volume'] < df['volume'].rolling(3).mean()) & \
                  ((df['close'] - df['low']) / (df['high'] - df['low']) < df['close'].rolling(3).mean() / df['high'].rolling(3).mean()) 
  
    # Identify the last Higher Low (HL) and Lower High (LH)
    
    df['higher_low'] = df['low'].rolling(window=30).min().shift(1)
    df['lower_high'] = df['high'].rolling(window=30).max().shift(1)
    df['bullish_choch'] = df['high'] > df['lower_high']
    df['bearish_choch'] = df['low'] < df['higher_low'] 

    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_decreasing'] = df['volume'].rolling(window=5).mean() < (df['volume'].rolling(window=10).mean())
    df['volume_spike'] = df['volume'] > (1.3 * df['vol_ma'])
    
    rolling_window = 50
    df['returns'] = df['close'].pct_change().fillna(0)
    df['expected_return'] = df['returns'].rolling(rolling_window).mean()
    df['variance'] = df['returns'].rolling(rolling_window).var()
    
    df['kelly_fraction'] = 0.5 * (df['expected_return'] / df['variance'])
    df['kelly_fraction'] = df['kelly_fraction'].fillna(0)  
    
#     # leverage
#     df['leverage'] = 1 + (0.5 * (1 - df['ATR'].rolling(rolling_window).rank(pct=True)))  
#     df['leverage'] = df['leverage'].clip(1, 2)
    
    # position-sizing
    df['atr_scaling'] = (df['ATR'] / df['ATR'].rolling(100).mean()).clip(0.5, 2)
    df['position'] = (0.5 + 0.5 * np.minimum(1, np.abs(df['kelly_fraction']))) * (1 / df['atr_scaling']) * 100
    df['position'] = df['position'].clip(80, 100)

    return df

# STRAT (SIGNAL GENERATION)
def strat(df):
    
    # Updated Trading Signals
    df['Signal_Long'] = (df['No_Supply'] & df['Go_Uptrend']) | (df['bullish_choch'] &  df['volume_spike']) | ((df['Trend'] == -1 & df['vol_decreasing']) & (df['Strong_Bullish_Next'] | df['Strong_Hammer']))
    df['Signal_Short'] = (df['No_Demand'] & df['Go_Downtrend']) | (df['bearish_choch'] &  df['volume_spike']) | ((df['Trend'] == 1 & df['vol_decreasing']) & (df['Strong_Bearish_Next'] | df['Strong_Inverted_Hammer']))
    
    df['signals'] = df.apply(lambda x: 1 if x['Signal_Long'] else 2 if x['Signal_Short'] else 0, axis=1)
    in_trade_long, in_trade_short = False, False
    df['remarks'] = 'Algo_Exit'

    for i in tqdm(range(len(df))):
        if df['signals'].iloc[i - 1] == -1:
            df.at[i, 'signals'] = 2
            in_trade_short = True
        if df['signals'].iloc[i - 1] == -2:
            df.at[i, 'signals'] = 1
            in_trade_long = True
        elif df['signals'].iloc[i] == 2 and in_trade_long:
            df.at[i, 'signals'] = -1
            in_trade_long = False
        elif df['signals'].iloc[i] == 1 and in_trade_short:
            df.at[i, 'signals'] = -2
            in_trade_short = False
        elif df['signals'].iloc[i] == 2 and not in_trade_short:
            in_trade_short = True
            in_trade_long = False
        elif df['signals'].iloc[i] == 1 and not in_trade_long:
            in_trade_long = True
            in_trade_short = False

    sl, entry_price, target_points = 0, 0, 0
    trailing_sl, intrade_long, intrade_short = 0, False, False

    df = identify_early_pullbacks(df)
    df = apply_smart_trailing_stop(df)

    for i in tqdm(range(len(df))):
        
        # *** RE-ENTRY
        
#         current_signal = df['signals'].iloc[i]
#         if current_signal == 0:  # No active trade
#             prev_signal = df['signals'].iloc[i - 1] if i > 0 else 0
#             prev_entry_price = df['HA_Close'].iloc[i - 1] if i > 0 else 0
            
#             if prev_signal == -1 and df["Trend"].iloc[i] == 1:
#                 df.at[i, 'signals'] = 1
#                 df.at[i, 'remarks'] = 'Reentry_Short'
            
#             elif prev_signal == -2 and df["Trend"].iloc[i] == -1:
#                 df.at[i, 'signals'] = 2
#                 df.at[i, 'remarks'] = 'Reentry_Long'

        # *** Stop-Loss
    
        if df['signals'].iloc[i] == 1:
            entry_price = df["HA_Close"].iloc[i]
            sl = calculate_stop_loss(df['high'].iloc[i], df['low'].iloc[i], entry_price, df['signals'].iloc[i]) - df['adaptive_multiplier'].iloc[i] * df['ATR'].iloc[i]
            trailing_sl = sl 
            scale_out_1, scale_out_2, final_exit = calculate_fibonacci_levels(entry_price, sl)
            intrade_long = True

        elif df['signals'].iloc[i] == 2:
            entry_price = df["HA_Close"].iloc[i]
            sl = calculate_stop_loss(df['high'].iloc[i], df['low'].iloc[i], entry_price, df['signals'].iloc[i]) + df['adaptive_multiplier'].iloc[i] * df['ATR'].iloc[i]
            trailing_sl = sl
            scale_out_1, scale_out_2, final_exit = calculate_fibonacci_levels(sl, entry_price)
            intrade_short = True

        elif intrade_short:
            if df['HA_Close'].iloc[i] >= sl:
                df['signals'].iloc[i] = -2
                df['remarks'].iloc[i] = 'SL_Exit'
                intrade_short = False
            elif df['HA_Close'].iloc[i] > entry_price and df['HA_Close'].iloc[i] < sl * 0.95:
                sl = entry_price
            else:
                trailing_sl = min(sl, df['smart_trailing_stop_short'].iloc[i])
                if df['HA_Close'].iloc[i] >= trailing_sl:
                    df['signals'].iloc[i] = -2
                    df['remarks'].iloc[i] = 'Trailing_SL_Exit'
                    intrade_short = False
                elif df['HA_Close'].iloc[i] <= scale_out_1:
                    df['remarks'].iloc[i] = 'Exited 50% at Scale Out 1'
                elif df['HA_Close'].iloc[i] <= scale_out_2:
                    df['remarks'].iloc[i] = 'Exited 25% at Scale Out 2'
                elif df['HA_Close'].iloc[i] <= final_exit:
                    df['signals'].iloc[i] = -2
                    df['remarks'].iloc[i] = 'Final Exit at Fibonacci Level'
                    intrade_short = False

        elif intrade_long:
            if df['HA_Close'].iloc[i] <= sl:
                df['signals'].iloc[i] = -1
                df['remarks'].iloc[i] = 'SL_Exit'
                intrade_long = False
            elif df['HA_Close'].iloc[i] < entry_price and df['HA_Close'].iloc[i] > sl * 1.05:
                sl = entry_price
            else:
                trailing_sl = max(sl, df['smart_trailing_stop_long'].iloc[i])
                if df['HA_Close'].iloc[i] <= trailing_sl:
                    df['signals'].iloc[i] = -1
                    df['remarks'].iloc[i] = 'Trailing_SL_Exit'
                    intrade_long = False
                elif df['HA_Close'].iloc[i] >= scale_out_1:
                    df['remarks'].iloc[i] = 'Exited 50% at Scale Out 1'
                elif df['HA_Close'].iloc[i] >= scale_out_2:
                    df['remarks'].iloc[i] = 'Exited 25% at Scale Out 2'
                elif df['HA_Close'].iloc[i] >= final_exit:
                    df['signals'].iloc[i] = -1
                    df['remarks'].iloc[i] = 'Final Exit at Fibonacci Level'
                    intrade_long = False

    df_1 = df
    long_open = False
    short_open = False

    for index, row in df_1.iterrows():
        signal = row['signals']

        if signal == 2 and short_open:
            df_1.at[index, 'signals'] = 0
        elif signal == 1 and long_open:
            df_1.at[index, 'signals'] = 0
        elif signal == 1 and long_open != True:
            long_open = True
        elif signal == 2 and short_open != True:
            short_open = True

        elif signal == -1:
            if long_open:
                long_open = False
            else:
                df_1.at[index, 'signals'] = 0
        elif signal == -2:
            if short_open:
                short_open = False
            else:
                df_1.at[index, 'signals'] = 0

    mapped_trade_type = dict(zip([1,-1,2,-2,0],['LONG', 'CLOSE', 'SHORT', 'CLOSE', 'HOLD']))
    df_1['trade_type'] = df_1['signals'].map(mapped_trade_type)
    
    df_1.to_csv('visualization_data.csv', index=False)

    df_1['signal'] = df_1['signals']
    dg = df_1[df_1['signal'] != 0]
    dg.to_csv('signals.csv')

    return df_1, dg

# executing process_data and strat
data = pd.read_csv('./ETH_Data/ETHUSDT_1d.csv')

data = process_data(data)
df, dg = strat(data)

##visualising signals
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Price')
plt.scatter(df[df['signals'] == 1].index, df[df['signals'] == 1]['close'], marker='^', color='g', label='Long Signal', alpha=1)
plt.scatter(df[df['signals'] == 2].index, df[df['signals'] == 2]['close'], marker='v', color='r', label='Short Signal', alpha=1)
plt.legend()
plt.show()

##  BACKTESTING USING UNTRADE SDK 
def update_signals_for_backtesting(data):
    dg = data
    for i in tqdm(range(len(dg))):
        if dg['signals'].iloc[i] == 2 :
            dg['signals'].iloc[i] = -1
        elif dg['signals'].iloc[i] == -2 :
            dg['signals'].iloc[i] = 1
    return dg

def perform_backtest(csv_file_path):
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="test",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,# Adjust leverage as needed
    )
    return result

if __name__ == "__main__":
    # Read data from CSV file
    data = pd.read_csv("signals.csv")
    data = data[data["signals"]!=0]
    data = data.rename(columns={'timestamp': 'datetime'})
    res1 = update_signals_for_backtesting(data)
    res = (res1)
    res.to_csv("processed_data.csv", index=False)

    # Perform backtest on processed data
    csv_file_path = "processed_data.csv"
    backtest_result = perform_backtest(csv_file_path)

    # Get the last value of backtest result
    last_value = None
    for value in backtest_result:
        # print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
        last_value = value
    print(last_value)