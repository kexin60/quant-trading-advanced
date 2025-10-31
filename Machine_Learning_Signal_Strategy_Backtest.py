import os, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='reports', help='output directory for reports')
parser.add_argument('--symbols', nargs='+', default=['AAPL','QQQ','VOO'], help='list of symbols to backtest')
args = parser.parse_args()

REPORT_DIR = args.outdir
os.makedirs(REPORT_DIR, exist_ok=True)

def load_price_file(sym_pattern):
    files = sorted(glob.glob(sym_pattern))
    if not files:
        return None, None
    df = pd.read_csv(files[0])
    cols = [c.lower() for c in df.columns]
    # find date and close
    date_col = None
    close_col = None
    for c in df.columns:
        if c.lower() in ['date','datetime','timestamp']:
            date_col = c
            break
    for c in df.columns:
        if 'close' in c.lower():
            close_col = c
            break
    if date_col is None or close_col is None:
        return None, None
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'Date', close_col: 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Date','Close'])
    df = df.groupby('Date').agg({'Close':'last'}).reset_index()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    return df, files[0]

def feature_engineer(df, factors_df=None):
    df = df.copy()
    # basic data sufficiency check
    if len(df) < 30:
        print("[WARN] Not enough data for feature engineering.")
        return df
    if factors_df is not None:
        df = df.merge(factors_df, on='Date', how='inner')
    # technical features matching earlier scripts
    df['MA5'] = df['Return'].rolling(5).mean()
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Momentum'] = df['Return'].shift(1)
    # Target: next day up
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna(subset=['MA5','Volatility','Momentum','Target'])
    return df

def compute_metrics(returns, rf=0.02):
    # returns: series of daily returns
    if len(returns) == 0:
        return {'ann_return':np.nan,'ann_vol':np.nan,'sharpe':np.nan,'max_drawdown':np.nan}
    TRADING_DAYS = 252
    cum = (1 + returns).prod()
    ann_return = cum ** (TRADING_DAYS / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()
    return {'ann_return':ann_return, 'ann_vol':ann_vol, 'sharpe':sharpe, 'max_drawdown':max_dd}

def backtest_signals(df, signal_series):
    # signal_series aligned with df rows, signal indicates prediction for next day (as in Target)
    # semantic, vectorized shift: position entered at close on day t and returns realized on t+1
    s = signal_series.copy()
    # ensure index alignment: use the DataFrame index (0..n-1) that callers pass
    strat_returns = df['Return'].shift(-1) * s
    # set meaningful index (dates) and drop missing tail
    strat_series = pd.Series(strat_returns.values, index=df['Date']).dropna()
    return strat_series

def load_factors():
    # try to load local factors if present (reuse earlier code's source if available)
    # fallback: no factors
    try:
        import requests, io, zipfile
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        r = requests.get(url, timeout=10)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.lower().endswith(('.csv','.txt'))][0]
        txt = z.read(name).decode('latin1')
        lines = txt.splitlines()
        header_idx = [i for i, ln in enumerate(lines) if 'Mkt' in ln][0]
        csv_text = '\n'.join(lines[header_idx:])
        fdf = pd.read_csv(io.StringIO(csv_text))
        fdf = fdf.rename(columns={fdf.columns[0]:'Date'})
        fdf['Date'] = pd.to_datetime(fdf['Date'].astype(str), format='%Y%m%d', errors='coerce')
        for c in ['MKT','SMB','HML','RF']:
            if c in fdf.columns:
                fdf[c] = fdf[c]/100
        return fdf[['Date','MKT','SMB','HML','RF']]
    except Exception:
        return None

def run_backtest():
    symbols = args.symbols
    factors = load_factors()
    results = []
    plt.figure(figsize=(10,6))
    for sym in symbols:
        df, path = load_price_file(f"{sym}*_data.csv")
        if df is None:
            print(f"Skipping {sym}: no data file")
            continue
        df_feat = feature_engineer(df.copy(), factors)
        features = ['MA5','Volatility','Momentum']
        if factors is not None:
            features += ['MKT','SMB','HML']
        X = df_feat[features]
        y = df_feat['Target']
        # small-sample guard
        if len(X) < 20:
            print(f"[WARN] Not enough rows after feature engineering for {sym} (n={len(X)}). Skipping.")
            continue
        # train/test split (no shuffle) - keep deterministic
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        # logistic
        log = LogisticRegression(max_iter=1000)
        log.fit(X_train_s, y_train)
        y_log = pd.Series(log.predict(X_test_s), index=X_test.index)
        # rf
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        rf.fit(X_train_s, y_train)
        y_rf = pd.Series(rf.predict(X_test_s), index=X_test.index)

        # construct test dataframe aligned to original df rows
        df_test = df_feat.loc[X_test.index].reset_index(drop=True)
        # align signals
        sig_log = pd.Series(y_log.values, index=df_test.index)
        sig_rf = pd.Series(y_rf.values, index=df_test.index)

        strat_log = backtest_signals(df_test, sig_log)
        strat_rf = backtest_signals(df_test, sig_rf)
        # benchmark buy and hold on test period
        bh = df_test['Return'].copy(); bh.index = df_test['Date']

        # metrics
        m_log = compute_metrics(strat_log.dropna())
        m_rf = compute_metrics(strat_rf.dropna())
        m_bh = compute_metrics(bh.iloc[1:])

        results.append({'symbol':sym, 'model':'logistic', **m_log})
        results.append({'symbol':sym, 'model':'random_forest', **m_rf})
        results.append({'symbol':sym, 'model':'buy_hold', **m_bh})

        # cumulative plot
        cum_log = (1 + strat_log.fillna(0)).cumprod() - 1
        cum_rf = (1 + strat_rf.fillna(0)).cumprod() - 1
        cum_bh = (1 + bh.fillna(0)).cumprod() - 1
        plt.plot(cum_log.index, cum_log.values, label=f'{sym} - LOG')
        plt.plot(cum_rf.index, cum_rf.values, label=f'{sym} - RF')
        plt.plot(cum_bh.index, cum_bh.values, label=f'{sym} - BH', linestyle=':')

        # save per-symbol CSVs
        out_df = pd.DataFrame({'date':cum_log.index, f'cum_log':cum_log.values, f'cum_rf':cum_rf.values, 'cum_bh':cum_bh.values})
        out_df.to_csv(os.path.join(REPORT_DIR, f'{sym}_strategy_cumrets.csv'), index=False)

    plt.legend(); plt.grid(True); plt.title('Cumulative Returns: Logistic vs Random Forest vs Buy & Hold'); plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'models_vs_bh_cumret.png'))
    plt.close()

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(REPORT_DIR, 'ml_signal_backtest_summary.csv'), index=False)
    # write markdown summary with aggregate stats and metadata
    with open(os.path.join(REPORT_DIR, 'ml_signal_backtest_report.md'), 'w', encoding='utf-8') as f:
        f.write('# ML Signal Strategy Backtest Summary\n\n')
        f.write('Comparison of LogisticRegression vs RandomForest and Buy & Hold.\n\n')
        f.write(res_df.to_markdown(index=False))
        # metadata
        try:
            f.write(f'\n- Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'- Total records used: {len(res_df)}\n')
        except Exception:
            pass
        # summary statistics
        try:
            f.write('\n\n## Summary Statistics\n')
            avg_sharpe_log = res_df[res_df.model == 'logistic']['sharpe'].mean()
            avg_sharpe_rf = res_df[res_df.model == 'random_forest']['sharpe'].mean()
            f.write(f'- Average Sharpe (LOG): {avg_sharpe_log:.2f}\n')
            f.write(f'- Average Sharpe (RF): {avg_sharpe_rf:.2f}\n')
        except Exception:
            # if res_df doesn't have expected columns, skip summary
            pass

    print('[INFO] Backtest complete. Reports saved under reports/')

if __name__ == '__main__':
    run_backtest()
