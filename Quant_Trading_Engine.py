# Trading+ System Integrated Script
# This script combines volatility forecasting, portfolio optimization, machine learning signals,
# and automated trading via the IBKR API (paper trading). Each section is modular with error handling and logging.
# Inline comments explain each part of the process for clarity.

import os
import math
import time
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# helper: compute performance metrics from a returns series (daily returns)
def compute_perf_stats(returns, rf=0.02):
    # returns: pandas Series of daily returns (can be empty)
    if returns is None or len(returns) == 0:
        return {'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 'max_drawdown': np.nan}
    TRADING_DAYS = 252
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 'max_drawdown': np.nan}
    cum = (1 + r).prod()
    ann_return = cum ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()
    return {'ann_return': ann_return, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_drawdown': max_dd}

# collect model summaries for the final report
model_summaries = []

# Import IBKR API (using ib_insync for asynchronous IB API interaction)
try:
    from ib_insync import IB, Stock, MarketOrder
except ImportError as e:
    raise ImportError("The ib_insync library is required for IBKR API interaction. Please install it.") from e

# Import ARCH library for GARCH volatility modeling
try:
    from arch import arch_model
except ImportError as e:
    raise ImportError("The arch library is required for GARCH volatility modeling. Please install it.") from e

# Import SciPy for portfolio optimization
try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError("The SciPy library is required for portfolio optimization. Please install it.") from e

# Configuration and parameters
parser = argparse.ArgumentParser()
parser.add_argument('--symbols', nargs='+', default=['QQQ','VOO','AAPL'], help='list of symbols to trade')
args = parser.parse_args()

SYMBOLS = args.symbols       # List of assets to trade
REFRESH_INTERVAL = 5                  # Market data refresh interval (seconds)
MARKET_DATA_TYPE = 3                  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
THRESHOLD_BUY = 0.65                  # If predicted upward probability >= this, signal a buy
THRESHOLD_SELL = 0.50                 # If predicted upward probability < this (and we hold position), signal a sell
MAX_WEIGHT = 0.4                      # Maximum portfolio weight for any single asset
ALLOW_SHORT = False                   # Allow short selling (False for long-only strategy)
MAX_POSITION = 20                     # Safety cap on position size (max number of shares per asset)
MIN_TRADE_INTERVAL = 60               # Minimum interval (seconds) between trades for the same asset (prevent rapid re-trading)
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))  # Starting capital for position sizing (for simulation or sizing logic)
IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
IB_PORT = int(os.getenv('IB_PORT', '7497'))   # Default TWS paper trading port
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))
SIMULATE = os.getenv('SIMULATE', '1').lower() not in ('0', 'false')  # Simulation mode (True by default for safety)
TEST_RUN_SECONDS = int(os.getenv('TEST_RUN_SECONDS', '0'))          # Optional: limit runtime duration for testing (0 means run indefinitely)
REPORT_DIR = 'reports'
os.makedirs(REPORT_DIR, exist_ok=True)

# ========== Data Retrieval and Preprocessing ==========
# Load historical price data for each symbol to generate returns and features
price_series = {}
for sym in SYMBOLS:
    # Find historical CSV file for the symbol (expects files like {SYM}_data.csv in current directory)
    files = [f for f in os.listdir('.') if f.startswith(sym) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No historical data file found for symbol {sym}.")
    # Prefer files containing 'data' (prioritize files with 'data_yf' if present)
    data_file = None
    for f in sorted(files):
        if 'data_yf' in f:
            data_file = f
            break
    if data_file is None:
        for f in sorted(files):
            if 'data' in f:
                data_file = f
                break
    if data_file is None:
        data_file = sorted(files)[0]
    # Load CSV data
    df = pd.read_csv(data_file)
    # Identify date and close price columns (case-insensitive)
    date_col = None
    price_col = None
    for c in df.columns:
        clower = c.lower()
        if clower in ['date', 'datetime', 'timestamp']:
            date_col = c
        if 'close' in clower or 'price' in clower:
            price_col = c
    if date_col is None or price_col is None:
        raise ValueError(f"Could not identify Date or Close price column in {data_file}.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    # Aggregate to daily frequency (if intraday data, take last price of each day)
    df = df.groupby(df[date_col].dt.date).agg({price_col: 'last'}).reset_index()
    df = df.rename(columns={date_col: 'Date', price_col: 'Close'})
    # Compute daily returns
    df['Return'] = df['Close'].pct_change()
    df = df.dropna(subset=['Return']).reset_index(drop=True)
    price_series[sym] = df[['Date', 'Close', 'Return']]

# Align returns for all symbols into one DataFrame on Date (inner join on common dates)
returns_df = None
for sym, sub_df in price_series.items():
    sub_df = sub_df.copy()
    sub_df.set_index('Date', inplace=True)
    series = sub_df['Return']
    if returns_df is None:
        returns_df = pd.DataFrame(series).rename(columns={'Return': sym})
    else:
        # Join on dates (inner join to require data for all symbols on a date)
        returns_df = returns_df.join(series.rename(sym), how='inner')
returns_df = returns_df.dropna()
if returns_df.empty:
    raise ValueError("Not enough overlapping historical data to compute returns for all symbols.")

# ========== Volatility Modeling & Risk Forecasting ==========
# Forecast volatility for each asset using a GARCH(1,1) model on recent returns
vol_forecast = {}  # one-step ahead volatility forecast (daily std dev)
lookback = 252     # number of recent trading days to use for volatility model (approx 1 year)
for sym in SYMBOLS:
    ret_series = returns_df[sym]
    if len(ret_series) < 10:
        # Too few data points for volatility model, fallback to sample std dev
        vol_forecast[sym] = ret_series.std()
        continue
    recent_returns = ret_series[-lookback:] if len(ret_series) > lookback else ret_series
    try:
        # Fit GARCH(1,1) on recent return series (scaled to percentages for stability)
        am = arch_model(recent_returns * 100, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
        res = am.fit(disp='off')
        # Forecast next period variance
        fcast = res.forecast(horizon=1, reindex=False)
        var_pred = fcast.variance.values[-1, 0]
        if np.isnan(var_pred):
            raise ValueError("GARCH forecast returned NaN")
        vol_forecast[sym] = math.sqrt(var_pred) / 100.0  # convert back to decimal form
    except Exception as e:
        # If GARCH fails, use sample volatility as fallback
        vol_forecast[sym] = recent_returns.std()
        print(f"[WARN] GARCH model failed for {sym}, using sample std dev: {vol_forecast[sym]:.4f}")

# ========== Multi-Asset Portfolio Optimization ==========
# Compute expected returns for each asset (using historical mean of recent returns as proxy)
returns_used = returns_df.copy()
if len(returns_used) > lookback:
    returns_used = returns_used.iloc[-lookback:]
exp_return = returns_used.mean()  # expected daily return for each asset
# Compute covariance matrix of returns and incorporate forecast volatilities for variance
cov_matrix = returns_used.cov()
for sym in SYMBOLS:
    cov_matrix.loc[sym, sym] = (vol_forecast.get(sym, returns_used[sym].std())) ** 2  # update variance to forecast
# Optimize portfolio weights to maximize Sharpe ratio (no shorting unless ALLOW_SHORT=True)
n = len(SYMBOLS)
initial_w = np.array([1.0/n] * n)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
# Set bounds for weights
if ALLOW_SHORT:
    lb = -MAX_WEIGHT  # allow limited shorting
else:
    lb = 0.0
bounds = [(lb, MAX_WEIGHT)] * n

def neg_sharpe(w):
    # Objective: negative Sharpe ratio = -(w^T exp_return) / sqrt(w^T Cov w)
    port_ret = np.dot(w, exp_return.values)
    port_var = float(np.dot(w, np.dot(cov_matrix.values, w.T)))
    port_vol = math.sqrt(port_var) if port_var > 0 else 0.0
    return 0 if port_vol == 0 else -(port_ret / port_vol)

opt_result = None
try:
    opt_result = minimize(neg_sharpe, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)
except Exception as e:
    print(f"[WARN] Portfolio optimization failed: {e}")
# Use optimized weights if successful, else equal weights
if opt_result is not None and opt_result.success:
    opt_w = opt_result.x
    if not ALLOW_SHORT:
        opt_w = np.clip(opt_w, 0, None)  # ensure no negative weight due to numerical errors
else:
    opt_w = initial_w
    print("[INFO] Using equal weights (optimization not successful).")

# Map optimized weights to symbols
target_weights = {sym: opt_w[i] for i, sym in enumerate(SYMBOLS)}
print("Target portfolio weights:", target_weights)

# ========== Machine Learning Signal Strategy (Model Training) ==========
# Train Logistic Regression and Random Forest on each asset's historical data, choose best model for each asset
models = {}
for sym in SYMBOLS:
    df = price_series[sym]  # DataFrame with Date, Close, Return for the symbol
    # Feature engineering: moving average, volatility, and momentum features from returns
    feat_df = pd.DataFrame({
        'MA5': df['Return'].rolling(window=5).mean(),
        'Volatility': df['Return'].rolling(window=10).std(),
        'Momentum': df['Return'].shift(1)
    })
    feat_df = feat_df.dropna()
    if feat_df.empty or len(feat_df) < 30:
        print(f"[WARN] Not enough data to train model for {sym}. Skipping.")
        continue
    # Prepare dataset for ML: features X and target y (1 if next day return > 0, else 0)
    target = df['Return'].shift(-1)  # next day's return (last will be NaN)
    common_idx = feat_df.index.intersection(target.dropna().index)
    df_ml = feat_df.loc[common_idx].copy()
    df_ml['target'] = (target.loc[common_idx] > 0).astype(int)
    X_all = df_ml[['MA5', 'Volatility', 'Momentum']]
    y_all = df_ml['target']
    # Train/test split (time-based, deterministic)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, shuffle=False, random_state=42)
    # Ensure sufficient training and test samples
    if len(X_train) < 20 or len(X_test) < 10:
        print(f"[WARN] Insufficient training data for {sym} (train={len(X_train)}, test={len(X_test)}). Skipping.")
        continue
    # Scale features
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    # Train Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_s, y_train)
    log_acc = accuracy_score(y_test, log_model.predict(X_test_s))
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf_model.fit(X_train_s, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_s))
    # Choose best model based on test accuracy
    if rf_acc >= log_acc:
        best_model = rf_model
        best_acc = rf_acc
        model_name = 'RandomForest'
    else:
        best_model = log_model
        best_acc = log_acc
        model_name = 'LogisticRegression'
    models[sym] = {'scaler': scaler, 'model': best_model}
    print(f"{sym}: Selected {model_name} model with test accuracy {best_acc:.2%}")

    # Compute simple backtest metrics on the test set (vectorized alignment)
    try:
        # align test rows to original df to get realized forward returns
        df_test = df.loc[X_test.index].copy()
        preds = best_model.predict(X_test_s)
        sig = pd.Series(preds, index=X_test.index)
        strat_returns = df_test['Return'].shift(-1) * sig
        stats = compute_perf_stats(strat_returns.dropna())
    except Exception:
        stats = {'ann_return':np.nan,'ann_vol':np.nan,'sharpe':np.nan,'max_drawdown':np.nan}

    model_summaries.append({'symbol': sym, 'model': model_name, 'test_accuracy': float(best_acc),
                            'ann_return_test': stats['ann_return'], 'sharpe_test': stats['sharpe'],
                            'max_drawdown_test': stats['max_drawdown']})

if not models:
    print("No models trained successfully. Exiting.")
    exit(1)

# ========== Interactive Brokers API Connection and Market Data Subscription ==========
ib = IB()
if not SIMULATE:
    # Confirm with user before placing real orders on paper account
    ans = input("SIMULATE mode is OFF (live trading). Proceed? (y/n): ").strip().lower()
    if ans != 'y':
        print("Aborting execution.")
        exit(0)
try:
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    ib.reqMarketDataType(MARKET_DATA_TYPE)
    print(f"Connected to IBKR (clientId={IB_CLIENT_ID}, host={IB_HOST}:{IB_PORT})")
except Exception as e:
    print(f"ERROR: Could not connect to IBKR API: {e}")
    exit(1)

# Define IB contracts and request market data for each symbol
contracts = {sym: Stock(sym, 'SMART', 'USD') for sym in SYMBOLS}
for contract in contracts.values():
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        print(f"[WARN] Could not qualify contract {contract}: {e}")
tickers = {sym: ib.reqMktData(contracts[sym], '', False, False) for sym in SYMBOLS}

# ========== Trade Execution Setup ==========
positions = defaultdict(int)               # current position (shares) for each symbol
entry_price = {sym: None for sym in SYMBOLS}  # entry price for current position (to compute PnL on exit)
last_trade_time = {sym: 0 for sym in SYMBOLS} # timestamp of last trade for each symbol
cash = INITIAL_CAPITAL                     # available cash for simulation (not used in live mode)

# If live, fetch existing positions (paper account might have open positions)
try:
    for p in ib.positions():
        sym = p.contract.symbol
        if sym in SYMBOLS:
            positions[sym] = int(p.position)
            if positions[sym] != 0:
                entry_price[sym] = None  # unknown entry price for pre-existing position
            print(f"[INFO] Starting with existing position: {sym} = {positions[sym]} shares")
except Exception as e:
    print(f"[WARN] Unable to fetch starting positions from IB: {e}")

# Prepare trade log file
log_filename = 'trade_log.csv'
if not os.path.exists(log_filename):
    with open(log_filename, 'w') as f:
        f.write('Time,Symbol,Probability,Prediction,Price,Action,Mode,PnL\n')

mode_flag = 'SIMULATE' if SIMULATE else 'LIVE'

# Helper: generate daily report (text and markdown) with PnL summary and equity curve
def generate_report():
    if not os.path.exists(log_filename):
        print("No trade log found, skipping report generation.")
        return
    df_log = pd.read_csv(log_filename)
    if df_log.empty:
        print("Trade log is empty, skipping report generation.")
        return
    if 'PnL' in df_log.columns:
        df_log['PnL'] = pd.to_numeric(df_log['PnL'], errors='coerce').fillna(0.0)
    else:
        df_log['PnL'] = 0.0
    df_log['Time'] = pd.to_datetime(df_log['Time'], errors='coerce')
    df_log = df_log.sort_values('Time').reset_index(drop=True)
    df_log['CumPnL'] = df_log['PnL'].cumsum()
    total_pnl = df_log['PnL'].sum()
    pnl_by_symbol = df_log.groupby('Symbol')['PnL'].sum().to_dict()
    wins = int((df_log['PnL'] > 0).sum())
    losses = int((df_log['PnL'] < 0).sum())
    # Write text summary
    txt_path = os.path.join(REPORT_DIR, 'trade_summary_report.txt')
    with open(txt_path, 'w') as f:
        f.write("===== Trading+ System Daily Report =====\n")
        f.write(f"Mode: {mode_flag}\n")
        f.write(f"Total Trades: {len(df_log)}\n")
        f.write(f"Total PnL: ${total_pnl:.2f}\n")
        for sym, pnl in pnl_by_symbol.items():
            f.write(f"{sym} PnL: ${pnl:.2f}\n")
        f.write(f"Wins: {wins} | Losses: {losses}\n")
        f.write("========================================\n")
    # Plot equity curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(df_log['Time'], df_log['CumPnL'], label='Cumulative PnL')
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Cumulative Returns: Logistic vs Random Forest vs Buy & Hold')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'equity_curve.png'))
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to generate equity curve plot: {e}")
    # Write markdown summary
    md_path = os.path.join(REPORT_DIR, 'trade_summary_report.md')
    with open(md_path, 'w') as f:
        f.write('# Daily Trading+ System Report\n\n')
        f.write(f'- Mode: {mode_flag}\n')
        # metadata: timestamp and total records
        try:
            f.write(f'- Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'- Total records used: {len(df_log)}\n')
        except Exception:
            pass
        f.write(f'- Total Trades: {len(df_log)}\n')
        f.write(f'- Total PnL: ${total_pnl:.2f}\n')
        for sym, pnl in pnl_by_symbol.items():
            f.write(f'- {sym} PnL: ${pnl:.2f}\n')
        f.write(f'- Wins: {wins} | Losses: {losses}\n\n')
        # performance metrics from trade log aggregated by day
        try:
            df_log['DateOnly'] = pd.to_datetime(df_log['Time']).dt.date
            daily_pnl = df_log.groupby('DateOnly')['PnL'].sum()
            daily_ret = (daily_pnl / float(INITIAL_CAPITAL)).astype(float)
            perf = compute_perf_stats(daily_ret)
            f.write('## Performance Metrics (based on trade log)\n')
            f.write(f'- Annualized Return: {perf["ann_return"]:.2%}\n')
            f.write(f'- Annualized Volatility: {perf["ann_vol"]:.2%}\n')
            f.write(f'- Sharpe (rf=2%): {perf["sharpe"]:.2f}\n')
            f.write(f'- Max Drawdown: {perf["max_drawdown"]:.2%}\n\n')
        except Exception:
            pass
        f.write('![Equity Curve](equity_curve.png)\n')
        # model summaries (from training)
        if model_summaries:
            f.write('\n## Model Summary (per symbol)\n')
            f.write('|Symbol|Model|Test Accuracy|Ann Return (test)|Sharpe (test)|Max Drawdown (test)|\n')
            f.write('|---:|:---:|:---:|:---:|:---:|:---:|\n')
            for m in model_summaries:
                f.write(f"|{m['symbol']}|{m['model']}|{m['test_accuracy']:.3f}|{(m['ann_return_test'] if m['ann_return_test'] is not None else np.nan):.2%}|{(m['sharpe_test'] if m['sharpe_test'] is not None else np.nan):.2f}|{(m['max_drawdown_test'] if m['max_drawdown_test'] is not None else np.nan):.2%}|\n")

# ========== Real-Time Signal Monitoring and Automated Trading Loop ==========
print("="*40)
print(f"[MODE] Trading Mode: {'SIMULATE (No orders will be sent)' if SIMULATE else 'LIVE (Orders will be executed!)'}")
print("="*40)
print("Starting real-time signal monitoring and trading...")
start_time = time.time()
try:
    while True:
        # If testing with time limit, break after TEST_RUN_SECONDS
        if TEST_RUN_SECONDS and (time.time() - start_time) > TEST_RUN_SECONDS:
            print(f"Test duration {TEST_RUN_SECONDS}s reached, stopping trading loop.")
            break
        ib.sleep(REFRESH_INTERVAL)  # wait for next market data refresh
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{time_str}] Market update:")
        for sym in SYMBOLS:
            ticker = tickers[sym]
            # Get latest price (wait a few seconds if data not immediately available)
            price = getattr(ticker, 'last', None) or getattr(ticker, 'close', None)
            waited = 0
            while price is None or math.isnan(price):
                if waited >= 5: break
                ib.sleep(1)
                waited += 1
                price = getattr(ticker, 'last', None) or getattr(ticker, 'close', None)
            if price is None or math.isnan(price):
                print(f"  {sym}: No price data available (skipping).")
                continue
            price = float(price)
            # Compute latest return and features using current price
            hist = price_series[sym].copy()
            last_close = float(hist['Close'].iloc[-1]) if len(hist) > 0 else price
            latest_return = (price / last_close - 1.0) if last_close != 0 else 0.0
            # Update recent returns list and calculate features
            recent_returns = list(hist['Return'].values)
            recent_returns.append(latest_return)
            ma5 = np.mean(recent_returns[-5:]) if len(recent_returns) >= 5 else np.mean(recent_returns)
            vol10 = np.std(recent_returns[-10:]) if len(recent_returns) >= 10 else np.std(recent_returns)
            momentum = recent_returns[-2] if len(recent_returns) >= 2 else 0.0  # yesterday's return
            # Predict using the trained model
            info = models.get(sym)
            if info is None:
                print(f"  {sym}: No model available, skipping.")
                continue
            X_now = np.array([[ma5, vol10, momentum]])
            X_now_s = info['scaler'].transform(X_now)
            model = info['model']
            if hasattr(model, 'predict_proba'):
                prob_up = float(model.predict_proba(X_now_s)[0, 1])
            else:
                # If model has no predict_proba (should not happen for Logistic/RandomForest)
                score = float(model.decision_function(X_now_s))
                prob_up = 1.0 / (1.0 + math.exp(-score))
            pred_label = 'Up' if prob_up >= 0.5 else 'Down'
            print(f"  {sym}: Price={price:.2f}, ProbUp={prob_up*100:.1f}%, Signal={pred_label}")
            # Enforce minimum trade interval for this symbol
            if time.time() - last_trade_time[sym] < MIN_TRADE_INTERVAL:
                # Too soon to trade again; log hold
                with open(log_filename, 'a') as f:
                    f.write(f"{time_str},{sym},{prob_up:.4f},{pred_label},{price:.2f},HOLD,{mode_flag},0.00\n")
                continue
            action = None
            trade_pnl = 0.0
            target_weight = target_weights.get(sym, 0)
            if prob_up >= THRESHOLD_BUY and positions[sym] == 0:
                # Buy if signal is strong and no current position
                if target_weight <= 0:
                    # Portfolio optimization suggests no allocation for this asset
                    print(f"  {sym}: Signal Up but target weight 0, skipping buy.")
                else:
                    # Determine quantity based on target weight and current portfolio value
                    portfolio_value = cash
                    for s2, shares in positions.items():
                        if shares > 0:
                            latest_price = getattr(tickers[s2], 'last', None) or getattr(tickers[s2], 'close', None)
                            if latest_price:
                                portfolio_value += float(latest_price) * shares
                    desired_value = target_weight * portfolio_value
                    qty = int(desired_value / price)
                    if qty > MAX_POSITION:
                        qty = MAX_POSITION
                    if qty <= 0:
                        # If calculation yields 0 shares (e.g., very small target allocation), skip
                        qty = 0
                    if qty > 0:
                        action = 'BUY'
                        if SIMULATE:
                            positions[sym] += qty
                            cash -= qty * price
                        else:
                            order = MarketOrder('BUY', qty)
                            trade = ib.placeOrder(contracts[sym], order)
                            ib.sleep(2)  # wait a moment for order to execute
                            print(f"    Order status: {trade.orderStatus.status}")
                            positions[sym] += qty
                        entry_price[sym] = price
                        last_trade_time[sym] = time.time()
                        print(f"  -> Executed BUY {qty} shares of {sym} at ${price:.2f}")
            elif prob_up < THRESHOLD_SELL and positions[sym] > 0:
                # Sell (close position) if probability drops below threshold and we are holding shares
                qty = positions[sym]
                action = 'SELL'
                if SIMULATE:
                    positions[sym] = 0
                    cash += qty * price
                else:
                    order = MarketOrder('SELL', qty)
                    trade = ib.placeOrder(contracts[sym], order)
                    ib.sleep(2)
                    print(f"    Order status: {trade.orderStatus.status}")
                    positions[sym] = 0
                # Calculate PnL for this trade if entry price known
                if entry_price[sym] is not None:
                    trade_pnl = (price - entry_price[sym]) * qty
                entry_price[sym] = None
                last_trade_time[sym] = time.time()
                print(f"  -> Executed SELL {qty} shares of {sym} at ${price:.2f}, PnL={trade_pnl:.2f}")
            else:
                # No trade (either already in position with no exit signal, or signal not strong enough to act)
                action = 'HOLD'
                trade_pnl = 0.0
            # Log the action and outcome for this symbol
            with open(log_filename, 'a') as f:
                f.write(f"{time_str},{sym},{prob_up:.4f},{pred_label},{price:.2f},{action},{mode_flag},{trade_pnl:.2f}\n")
except KeyboardInterrupt:
    print("\nManual interruption received. Exiting trading loop...")
finally:
    # On exit, generate report and disconnect
    try:
        generate_report()
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
    ib.disconnect()
    print("Disconnected from IBKR. Program terminated.")
