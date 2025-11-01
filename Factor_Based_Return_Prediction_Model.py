import os
import glob
import argparse
from datetime import datetime
import warnings
import json
import concurrent.futures
import joblib
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

REPORT_DIR = 'reports'
os.makedirs(REPORT_DIR, exist_ok=True)


def fetch_fama_french_daily():
    """Try to download daily Fama-French factors. Return DataFrame with Date, MKT, SMB, HML, RF (as decimals)"""
    try:
        import requests, io, zipfile
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        r = requests.get(url, timeout=10)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.lower().endswith(('.csv', '.txt'))][0]
        txt = z.read(name).decode('latin1')
        lines = txt.splitlines()
        header_idx = [i for i, ln in enumerate(lines) if 'Mkt-RF' in ln or 'Mkt' in ln]
        if not header_idx:
            return None
        header_idx = header_idx[0]
        csv_text = '\n'.join(lines[header_idx:])
        fdf = pd.read_csv(io.StringIO(csv_text))
        fdf = fdf.rename(columns={fdf.columns[0]: 'Date'})
        # Date format like YYYYMMDD
        fdf['Date'] = pd.to_datetime(fdf['Date'].astype(str), format='%Y%m%d', errors='coerce')
        # normalize percent columns if present
        for c in fdf.columns:
            if c not in ['Date']:
                try:
                    fdf[c] = pd.to_numeric(fdf[c], errors='coerce')
                except Exception:
                    pass
        # If Mkt-RF present, compute MKT = Mkt-RF + RF
        cols = [c.lower() for c in fdf.columns]
        if 'mkt-rf' in cols and 'rf' in cols:
            # standardize column names
            fdf = fdf.rename(columns={fdf.columns[cols.index('mkt-rf')]: 'MKT-RF', fdf.columns[cols.index('rf')]: 'RF'})
            fdf['MKT'] = fdf['MKT-RF'] + fdf['RF']
        # convert percentages to decimals if values look like percents (>1)
        for c in ['MKT', 'SMB', 'HML', 'RF']:
            if c in fdf.columns:
                if fdf[c].abs().median() > 1:
                    fdf[c] = fdf[c] / 100.0
        return fdf[['Date'] + [c for c in ['MKT', 'SMB', 'HML', 'RF'] if c in fdf.columns]]
    except Exception:
        return None


def load_price(symbol_pattern):
    files = sorted(glob.glob(symbol_pattern))
    if not files:
        return None
    df = pd.read_csv(files[0])
    # find date and close
    date_col = None
    close_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ['date', 'datetime', 'timestamp']:
            date_col = c
        if 'close' in lc or 'price' in lc:
            close_col = c
    if date_col is None or close_col is None:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'Date', close_col: 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.sort_values('Date').dropna(subset=['Date', 'Close']).reset_index(drop=True)
    df['Return'] = df['Close'].pct_change()
    df = df.dropna(subset=['Return']).reset_index(drop=True)
    return df


def build_features(df, ff_df=None, sentiment_df=None, mom_window=63, vol_window=21, use_log=False, beta_window=63):
    # assumes df has Date, Close, Return
    data = df[['Date', 'Close', 'Return']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    # optional log return
    if use_log:
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Return'] = data['log_return']
    data = data.set_index('Date')
    # momentum: past mom_window returns (simple cumulative)
    data[f'mom_{mom_window}'] = (1 + data['Return']).rolling(window=mom_window).apply(np.prod, raw=True) - 1
    # volatility: rolling std
    data[f'vol_{vol_window}'] = data['Return'].rolling(window=vol_window).std()
    # sentiment: if provided, merge by Date; else compute price-based proxy: rolling skew
    if sentiment_df is not None:
        s = sentiment_df.copy()
        s['Date'] = pd.to_datetime(s['Date'])
        s = s.set_index('Date')
        data = data.join(s[['Sentiment']], how='left')
    else:
        data['sentiment_proxy'] = data['Return'].rolling(window=vol_window).skew()

    # Merge Fama-French if available
    if ff_df is not None:
        ff = ff_df.copy()
        ff['Date'] = pd.to_datetime(ff['Date'])
        ff = ff.set_index('Date')
        data = data.join(ff, how='left')

    # Beta: rolling covariance with market (use ff_df MKT if available)
    if ff_df is not None and 'MKT' in ff_df.columns:
        ff = ff_df.copy()
        ff['Date'] = pd.to_datetime(ff['Date'])
        ff = ff.set_index('Date')
        mkt = ff['MKT'].reindex(data.index).fillna(method='ffill')
        cov = data['Return'].rolling(window=beta_window).cov(mkt)
        var = mkt.rolling(window=beta_window).var()
        data[f'beta_{beta_window}'] = cov / var
    else:
        data[f'beta_{beta_window}'] = np.nan

    # Reversal factor: negative past 5-day mean
    data['reversal_5'] = - data['Return'].rolling(window=5).mean()

    # Liquidity proxy: price change magnitude divided by volume if available
    vol_col = None
    for c in df.columns:
        if c.lower() == 'volume':
            vol_col = c
            break
    if vol_col is not None:
        vol_ser = pd.to_numeric(df[vol_col], errors='coerce')
        vol_ser.index = pd.to_datetime(df['Date'])
        vol_ser = vol_ser.reindex(data.index)
        data['liquidity_proxy'] = data['Return'].abs() / vol_ser.replace(0, np.nan)
    else:
        data['liquidity_proxy'] = np.nan

    # target: next-day excess return over RF (approx RF/252)
    if 'RF' in data.columns:
        data['RF_daily'] = data['RF'] / 252.0
    else:
        data['RF_daily'] = 0.0
    data['excess_ret'] = data['Return'].shift(-1) - data['RF_daily']

    # missing value handling: forward-fill then drop remaining NaNs
    data = data.fillna(method='ffill').dropna()
    return data


def fit_and_report(data, features, model_type='ridge', random_state=42, out_prefix='model', tscv_splits=0, alphas=None):
    X = data[features].copy()
    y = data['excess_ret'].copy()
    # If TimeSeriesSplit is requested, use the last fold as the final test set
    cv_summary = None
    if tscv_splits and tscv_splits > 1:
        tscv = TimeSeriesSplit(n_splits=tscv_splits)
        cv_metrics_list = []
        cv_dir_acc_list = []
        last_train_idx = None
        last_test_idx = None
        for tr_idx, te_idx in tscv.split(X):
            last_train_idx, last_test_idx = tr_idx, te_idx
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            scaler_cv = StandardScaler().fit(Xtr)
            Xtr_s = scaler_cv.transform(Xtr)
            Xte_s = scaler_cv.transform(Xte)

            # fit model per fold (simple version — metrics only)
            if model_type.lower() in ['ols', 'linear', 'lr']:
                m = LinearRegression()
                m.fit(Xtr_s, ytr)
            elif model_type.lower() in ['lasso', 'lassocv']:
                # LassoCV default inside CV fold
                m = LassoCV(cv=3, random_state=random_state).fit(Xtr_s, ytr)
            elif model_type.lower() in ['ridge', 'ridgecv']:
                m = RidgeCV(alphas=np.logspace(-4, 4, 50)).fit(Xtr_s, ytr)
            elif model_type.lower() in ['rf', 'randomforest']:
                m = RandomForestRegressor(n_estimators=200, random_state=random_state)
                m.fit(Xtr_s, ytr)
            elif model_type.lower() in ['xgb', 'xgboost'] and XGBOOST_AVAILABLE:
                m = xgb.XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0)
                m.fit(Xtr_s, ytr)
            elif model_type.lower() in ['lgb', 'lightgbm'] and LIGHTGBM_AVAILABLE:
                m = lgb.LGBMRegressor(n_estimators=200, random_state=random_state)
                m.fit(Xtr_s, ytr)
            else:
                raise ValueError(f'Model {model_type} not supported or required package missing')

            yte_pred = pd.Series(m.predict(Xte_s), index=yte.index)
            m_met = {
                'mae': mean_absolute_error(yte, yte_pred),
                'rmse': mean_squared_error(yte, yte_pred, squared=False),
                'r2': r2_score(yte, yte_pred)
            }
            cv_metrics_list.append(m_met)
            cv_dir_acc_list.append(float((np.sign(yte) == np.sign(yte_pred)).mean()))

        # aggregate CV metrics
        cv_mean = {
            'mae': float(np.mean([m['mae'] for m in cv_metrics_list])),
            'rmse': float(np.mean([m['rmse'] for m in cv_metrics_list])),
            'r2': float(np.mean([m['r2'] for m in cv_metrics_list]))
        }
        cv_summary = {
            'n_splits': tscv_splits,
            'cv_mean_metrics': cv_mean,
            'cv_dir_acc_mean': float(np.mean(cv_dir_acc_list))
        }

        # set final train/test to last fold
        X_train, X_test = X.iloc[last_train_idx], X.iloc[last_test_idx]
        y_train, y_test = y.iloc[last_train_idx], y.iloc[last_test_idx]
    else:
        # default 70/30 time-based split
        split = int(0.7 * len(X))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = None
    coefs = None
    feature_importances = None

    if model_type.lower() in ['ols', 'linear', 'lr']:
        model = LinearRegression()
        model.fit(X_train_s, y_train)
        coefs = model.coef_
    elif model_type.lower() in ['lasso', 'lassocv']:
        # use cross-validated LASSO; accept alphas if provided
        if alphas is not None:
            model = LassoCV(alphas=alphas, cv=5, random_state=random_state).fit(X_train_s, y_train)
        else:
            model = LassoCV(cv=5, random_state=random_state).fit(X_train_s, y_train)
        coefs = model.coef_
    elif model_type.lower() in ['ridge', 'ridgecv']:
        if alphas is not None:
            model = RidgeCV(alphas=alphas).fit(X_train_s, y_train)
        else:
            model = RidgeCV(alphas=np.logspace(-4, 4, 50)).fit(X_train_s, y_train)
        coefs = model.coef_
    elif model_type.lower() in ['rf', 'randomforest']:
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        model.fit(X_train_s, y_train)
        feature_importances = model.feature_importances_
    elif model_type.lower() in ['xgb', 'xgboost'] and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0)
        model.fit(X_train_s, y_train)
        try:
            feature_importances = model.feature_importances_
        except Exception:
            feature_importances = None
    elif model_type.lower() in ['lgb', 'lightgbm'] and LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(n_estimators=200, random_state=random_state)
        model.fit(X_train_s, y_train)
        feature_importances = model.feature_importances_
    else:
        raise ValueError(f'Model {model_type} not supported or required package missing')

    # Predictions
    y_pred_train = pd.Series(model.predict(X_train_s), index=y_train.index)
    y_pred_test = pd.Series(model.predict(X_test_s), index=y_test.index)

    # Directional accuracy (hit rate)
    def directional_accuracy(y_true, y_pred):
        # treat zero as no movement; compare signs
        return float((np.sign(y_true) == np.sign(y_pred)).mean())

    dir_acc_train = directional_accuracy(y_train, y_pred_train)
    dir_acc_test = directional_accuracy(y_test, y_pred_test)

    

    # Metrics
    def metrics(y_true, y_pred):
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'r2': r2_score(y_true, y_pred)
        }

    m_train = metrics(y_train, y_pred_train)
    m_test = metrics(y_test, y_pred_test)

    # Metrics heatmap for quick visual comparison
    try:
        metrics_df = pd.DataFrame([m_train, m_test], index=['Train', 'Test'])
        plt.figure(figsize=(6, 3))
        sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='vlag')
        plt.title(f"{out_prefix} performance summary")
        heat_path = os.path.join(REPORT_DIR, f'{out_prefix}_metrics_heatmap.png')
        plt.tight_layout()
        plt.savefig(heat_path)
        plt.close()
    except Exception:
        heat_path = None

    # Exposure plot: coefficients or importances
    labels = features
    plt.figure(figsize=(8, 4))
    if coefs is not None:
        vals = coefs
        plt.bar(labels, vals, color='tab:blue')
        plt.ylabel('Coefficient')
        title = f'Factor exposures ({model_type})'
    elif feature_importances is not None:
        vals = feature_importances
        plt.bar(labels, vals, color='tab:orange')
        plt.ylabel('Feature importance')
        title = f'Feature importances ({model_type})'
    else:
        plt.text(0.5, 0.5, 'No exposures available for this model', ha='center')
        title = f'Exposures ({model_type})'
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    exp_path = os.path.join(REPORT_DIR, f'{out_prefix}_factor_exposures.png')
    plt.savefig(exp_path)
    plt.close()

    # Predicted returns distribution (test set)
    plt.figure(figsize=(8, 4))
    plt.hist(y_pred_test, bins=50, density=True, alpha=0.6, label='Predicted')
    plt.hist(y_test, bins=50, density=True, alpha=0.4, label='Actual')
    plt.legend()
    plt.title(f'Predicted vs Actual excess return distribution ({model_type})')
    plt.tight_layout()
    dist_path = os.path.join(REPORT_DIR, f'{out_prefix}_predicted_returns_distribution.png')
    plt.savefig(dist_path)
    plt.close()

    # Save predictions
    preds_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred_test})
    preds_csv = os.path.join(REPORT_DIR, f'{out_prefix}_predictions.csv')
    preds_df.to_csv(preds_csv)

    # Write a short markdown report
    md_path = os.path.join(REPORT_DIR, f'{out_prefix}_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Factor-Based Return Prediction Report\n\n')
        f.write(f'- Model: {model_type}\n')
        f.write(f'- Trained on: {y_train.index.min().date()} -> {y_train.index.max().date()}\n')
        f.write(f'- Tested on: {y_test.index.min().date()} -> {y_test.index.max().date()}\n')
        f.write('\n## Metrics\n')
        f.write('### Train\n')
        for k, v in m_train.items():
            f.write(f'- {k}: {v:.6f}\n')
        f.write('### Test\n')
        for k, v in m_test.items():
            f.write(f'- {k}: {v:.6f}\n')
        # directional accuracy
        f.write(f'\n- Directional accuracy (train): {dir_acc_train:.4f}\n')
        f.write(f'- Directional accuracy (test): {dir_acc_test:.4f}\n')
        if cv_summary is not None:
            f.write('\n## Cross-validation summary\n')
            f.write(f'- n_splits: {cv_summary["n_splits"]}\n')
            for k, v in cv_summary['cv_mean_metrics'].items():
                f.write(f'- CV mean {k}: {v:.6f}\n')
            f.write(f'- CV mean directional accuracy: {cv_summary["cv_dir_acc_mean"]:.4f}\n')
        f.write('\n## Outputs\n')
        f.write(f'- Exposures plot: {os.path.basename(exp_path)}\n')
        f.write(f'- Predicted distribution: {os.path.basename(dist_path)}\n')
        f.write(f'- Predictions CSV: {os.path.basename(preds_csv)}\n')
        if heat_path is not None:
            f.write(f'- Metrics heatmap: {os.path.basename(heat_path)}\n')

        # Top factor contributions
        try:
            if coefs is not None:
                top_idx = np.argsort(np.abs(coefs))[-5:][::-1]
                top_features = [labels[i] for i in top_idx]
            elif feature_importances is not None:
                top_idx = np.argsort(feature_importances)[-5:][::-1]
                top_features = [labels[i] for i in top_idx]
            else:
                top_features = []
            if top_features:
                f.write(f"\nTop factors: {', '.join(top_features)}\n")
        except Exception:
            top_features = []

    summary = {
        'model': model_type,
        'train_metrics': m_train,
        'test_metrics': m_test,
        'dir_acc_train': dir_acc_train,
        'dir_acc_test': dir_acc_test,
        'cv_summary': cv_summary,
        'exposures_path': exp_path,
        'dist_path': dist_path,
        'preds_csv': preds_csv
    }
    # Save model and scaler for reproducibility
    try:
        model_path = os.path.join(REPORT_DIR, f'{out_prefix}_{model_type}.pkl')
        scaler_path = os.path.join(REPORT_DIR, f'{out_prefix}_scaler.pkl')
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        summary['model_path'] = model_path
        summary['scaler_path'] = scaler_path
    except Exception:
        summary['model_path'] = None
        summary['scaler_path'] = None

    # add heatmap and top_features info
    summary['heatmap_path'] = heat_path if 'heat_path' in locals() else None
    summary['top_features'] = top_features if 'top_features' in locals() else []

    return model, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', nargs='+', default=['AAPL'], help='symbols to model (will look for {SYM}*_data.csv)')
    p.add_argument('--model', type=str, default='ridge', help='model type: ols|lasso|ridge|rf|xgb|lgb')
    p.add_argument('--sentiment-csv', type=str, default=None, help='optional sentiment CSV with Date,Sentiment')
    p.add_argument('--outprefix', type=str, default='factor_model', help='output filename prefix')
    p.add_argument('--use-log', action='store_true', help='use log returns instead of simple pct returns')
    p.add_argument('--beta-window', type=int, default=63, help='rolling window for beta in days')
    p.add_argument('--mom-window', type=int, default=63, help='momentum window in days')
    p.add_argument('--vol-window', type=int, default=21, help='volatility window in days')
    p.add_argument('--tscv-splits', type=int, default=5, help='number of TimeSeriesSplit folds for CV (0 to disable)')
    p.add_argument('--alphas', type=str, default=None, help='comma-separated alphas or start:stop:count for logspace (e.g. 1e-4:1e4:50 or 0.001,0.01)')
    args = p.parse_args()

    # If config.yaml exists and yaml available, override args with config values
    if YAML_AVAILABLE and os.path.exists('config.yaml'):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as fh:
                cfg = yaml.safe_load(fh)
            if isinstance(cfg, dict):
                # map known keys
                for k in ['model', 'use_log', 'beta_window', 'mom_window', 'vol_window', 'tscv_splits', 'alphas', 'symbols']:
                    if k in cfg:
                        setattr(args, k if k != 'use_log' else 'use_log', cfg[k])
        except Exception:
            pass

    # parse alphas string into numeric list if provided
    def parse_alphas(s):
        if s is None:
            return None
        s = str(s)
        if ':' in s:
            # start:stop:count (logspace)
            parts = s.split(':')
            if len(parts) == 3:
                start = float(parts[0])
                stop = float(parts[1])
                count = int(parts[2])
                return np.logspace(np.log10(start), np.log10(stop), count)
        # comma separated
        try:
            return [float(x) for x in s.split(',') if x.strip()!='']
        except Exception:
            return None

    alphas_parsed = parse_alphas(args.alphas)

    ff = fetch_fama_french_daily()
    if ff is None:
        print('[WARN] Could not fetch Fama-French daily factors. Continuing without FF factors.')

    sentiment_df = None
    if args.sentiment_csv and os.path.exists(args.sentiment_csv):
        sd = pd.read_csv(args.sentiment_csv)
        if 'Date' in sd.columns and 'Sentiment' in sd.columns:
            sentiment_df = sd
        else:
            print('[WARN] sentiment CSV must contain Date and Sentiment columns. Ignoring sentiment file.')

    all_summaries = []
    for sym in args.symbols:
        pattern = f"{sym}*_data.csv"
        df = load_price(pattern)
        if df is None:
            print(f"[WARN] No price file found for {sym} (pattern {pattern}). Skipping.")
            continue
        data = build_features(df, ff_df=ff, sentiment_df=sentiment_df,
                              mom_window=args.mom_window, vol_window=args.vol_window,
                              use_log=args.use_log, beta_window=args.beta_window)
        if data.shape[0] < 100:
            print(f"[WARN] Not enough rows after feature construction for {sym} (n={len(data)}). Need ~100+. Skipping.")
            continue
        # candidate features (drop raw Close/Return)
        features = [c for c in data.columns if c not in ['Close', 'Return', 'excess_ret', 'RF_daily']]
        # ensure no object columns
        features = [c for c in features if np.issubdtype(data[c].dtype, np.number)]
        model, summary = fit_and_report(data, features, model_type=args.model, out_prefix=f"{args.outprefix}_{sym}", tscv_splits=args.tscv_splits, alphas=alphas_parsed)
        all_summaries.append((sym, summary))

    # aggregate short summary
    md_all = os.path.join(REPORT_DIR, f'{args.outprefix}_summary.md')
    with open(md_all, 'w', encoding='utf-8') as f:
        f.write('# Factor Model Run Summary\n\n')
        f.write(f'- Symbols: {args.symbols}\n')
        f.write(f'- Model: {args.model}\n')
        f.write(f'- Run at: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
        for sym, s in all_summaries:
            f.write(f'## {sym}\n')
            f.write(f'- Train MAE: {s["train_metrics"]["mae"]:.6f} | Test MAE: {s["test_metrics"]["mae"]:.6f}\n')
            f.write(f'- Train RMSE: {s["train_metrics"]["rmse"]:.6f} | Test RMSE: {s["test_metrics"]["rmse"]:.6f}\n')
            f.write(f'- Test R2: {s["test_metrics"]["r2"]:.4f}\n')
            f.write(f'- Exposures plot: {os.path.basename(s["exposures_path"])}\n')
            f.write(f'- Predicted distribution: {os.path.basename(s["dist_path"])}\n\n')

    print('[INFO] Finished. Reports saved under reports/')


if __name__ == '__main__':
    main()
