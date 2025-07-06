import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from tqdm import tqdm
from scipy.stats import linregress
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

def compute_mds(df, price_col='close', volume_col='volume', roc_period=10, norm_window=50):
    price_momentum = df[price_col].pct_change(roc_period)
    volume_momentum = df[volume_col].pct_change(roc_period)
    mds = price_momentum - volume_momentum

    rolling_min = mds.rolling(window=norm_window).min()
    rolling_max = mds.rolling(window=norm_window).max()
    df['MDS'] = ((mds - rolling_min) / (rolling_max - rolling_min))*2 - 1
    return df

def donchian_breakout(df, lookback=30, base_x=0.03, vol_window=10, min_x=0.02, max_x=0.04):
    # Step 1: Donchian channels
    df['dcUpper'] = df['high'].rolling(window=lookback).max().shift(1)
    df['dcLower'] = df['low'].rolling(window=lookback).min().shift(1)

    # Step 2: Rolling volume z-score
    df['vol_mean'] = df['volume'].rolling(vol_window).mean()
    df['vol_std'] = df['volume'].rolling(vol_window).std()
    df['vol_zscore'] = (df['volume'] - df['vol_mean']) / (df['vol_std'] + 1e-9)  # avoid divide-by-zero

    # Step 3: Normalize z-score to 0–1 using sigmoid, then scale x
    df['vol_sigmoid'] = 1 / (1 + np.exp(-df['vol_zscore']))
    df['x'] = base_x * df['vol_sigmoid']  # adaptive x

    # Clip x to prevent extreme thresholds
    df['x'] = df['x'].clip(lower=min_x, upper=max_x)
    
    df['dcUpper_modified'] = df['dcUpper'] * (1 - df['x'])
    df['dcLower_modified'] = df['dcLower'] * (1 + df['x'])

    # Step 4: Adaptive Donchian Breakout Signals
    df['donch'] = 0
    df['donch'] = np.where(df['close'] > df['dcUpper_modified'], 1, df['donch'])
    df['donch'] = np.where(df['close'] < df['dcLower_modified'], -1, df['donch'])
    return df

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
        close = df['close'].iloc[i]
        atr = df['ATR'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        open_ = df['open'].iloc[i]
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
            new_long_stop = min(close, open_) - (atr_mult * atr)
            if active_long_stop is None or new_long_stop > active_long_stop:
                active_long_stop = new_long_stop
        
        # Predict pullback for short positions
        if df['pullback_downtrend'].iloc[i]:
            new_short_stop = max(close, open_) + (atr_mult * atr)
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
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    df = donchian_breakout(df)
#     df = compute_mds(df)
    df = calculate_trend(df)

    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_decreasing'] = df['volume'].rolling(window=5).mean() < (df['volume'].rolling(window=10).mean())
    df['volume_spike'] = df['volume'] > (1.3 * df['vol_ma'])
    
    # Calculate relative volume
    df['avg_volume'] = df['volume'].rolling(window=10).mean()
    df['relative_volume'] = df['volume'] / df['avg_volume']

    # Adaptive multiplier based on relative volume
    df['adaptive_multiplier'] = np.where(df['relative_volume'] > 1.5, 1.0,    # High volume
                                     np.where(df['relative_volume'] < 0.7, 0.6,  # Low volume
                                              0.8))  # Medium volume
    
    rolling_window = 50
    df['returns'] = df['close'].pct_change().fillna(0)
    df['expected_return'] = df['returns'].rolling(rolling_window).mean()
    df['variance'] = df['returns'].rolling(rolling_window).var()
    
    df['kelly_fraction'] = 0.5 * (df['expected_return'] / df['variance'])
    df['kelly_fraction'] = df['kelly_fraction'].fillna(0)  
    
#     # leverage
#     df['leverage'] = 1 + (0.5 * (1 - df['ATR'].rolling(rolling_window).rank(pct=True)))  
#     df['leverage'] = df['leverage'].clip(1, 2)
    
#     # position-sizing
    df['atr_scaling'] = (df['ATR'] / df['ATR'].rolling(100).mean()).clip(0.5, 2)
    df['position'] = (0.5 + 0.5 * np.minimum(1, np.abs(df['kelly_fraction']))) * (1 / df['atr_scaling']) * 100
    df['position'] = df['position'].clip(60, 100)

    return df

# SIGNAL GENERATION
def strat(df):
    
    # Updated Trading Signals
    df['Signal_Long'] = (df['donch'] == 1) #& (df['MDS'] > 0.2)
    df['Signal_Short'] = (df['donch'] == -1) #& (df['MDS'] < -0.2)
    
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
#                 df.at[i, 'remarks'] = 'Reentry_Long'
            
#             elif prev_signal == -2 and df["Trend"].iloc[i] == -1:
#                 df.at[i, 'signals'] = 2
#                 df.at[i, 'remarks'] = 'Reentry_Short'

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