"""
Volatility Modeling & Risk Forecasting
- Fit GARCH(1,1) on each asset, produce one-step-ahead rolling volatility forecasts
- Compute VaR and ES (normal or Student-t) from model forecasts
- Compare realized vs predicted volatility and save plots/reports

Outputs per asset (under ./reports/):
- {TICKER}_vol_forecast.csv
- {TICKER}_vol_forecast.png
- {TICKER}_VaR_ES.csv
- volatility_risk_report_{TICKER}.md

Usage: python "Volatility Modeling & Risk Forecasting.py" --files-glob "*_data.csv"
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t

try:
    from arch import arch_model
except Exception:
    raise RuntimeError('arch 库未安装，请先 pip install arch')

parser = argparse.ArgumentParser()
parser.add_argument('--files-glob', type=str, default='*_data.csv')
parser.add_argument('--outdir', type=str, default='reports')
parser.add_argument('--lookback-days', type=int, default=252)
parser.add_argument('--step', type=int, default=1, help='步长（天）用于滚动预测')
parser.add_argument('--alpha-list', type=float, nargs='+', default=[0.01, 0.05],
                    help='VaR/ES 置信水平列表，例如 --alpha-list 0.01 0.05')
parser.add_argument('--dist', type=str, choices=['normal','t','auto'], default='auto',
                    help='当 auto 时，使用模型 fit 时的残差分布')
parser.add_argument('--rolling-refit', action='store_true', help='Perform true rolling refit forecasts (slow)')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# discover files
FILES = sorted(glob.glob(args.files_glob))
if not FILES:
    raise FileNotFoundError('未找到数据文件，请在当前目录放置 *_data.csv 或修改 --files-glob')

def fit_garch_series(returns, vol_lookback):
    """Fit GARCH(1,1) on a rolling basis and forecast 1-step ahead variance.
    returns: pandas Series indexed by date
    vol_lookback: number of obs used for each window
    Returns DataFrame with columns: date, realized_ret, pred_sigma, roll_realized_vol
    """
    # Faster approach: fit GARCH once on full series and use one-step shifted conditional volatility
    r = returns.dropna()
    if len(r) < vol_lookback:
        raise ValueError('样本长度小于 lookback_days，无法拟合')
    pred_sigma = pd.Series(index=r.index, data=[np.nan]*len(r))
    nu = None
    model_info = {'aic': None, 'bic': None}
    if args.rolling_refit:
        # True rolling refit (slower): re-fit on each window and forecast one-step ahead
        for i in range(vol_lookback, len(r)):
            sub = r.iloc[i-vol_lookback:i]
            try:
                am = arch_model(sub, vol='Garch', p=1, q=1, mean='Constant', dist='StudentsT', rescale=False)
                res = am.fit(disp='off')
                f = res.forecast(horizon=1, reindex=False)
                try:
                    var1 = float(f.variance.values[-1, 0])
                except Exception:
                    var1 = float(res.conditional_volatility.iloc[-1] ** 2)
                pred_sigma.iloc[i] = np.sqrt(max(var1, 0.0))
                # record last fit info
                try:
                    model_info['aic'] = float(res.aic)
                    model_info['bic'] = float(res.bic)
                except Exception:
                    pass
                try:
                    if hasattr(res, 'params') and 'nu' in res.params.index:
                        nu = float(res.params['nu'])
                except Exception:
                    pass
            except Exception:
                # leave NaN
                continue
    else:
        try:
            # disable automatic rescaling warnings by setting rescale=False
            am = arch_model(r, vol='Garch', p=1, q=1, mean='Constant', dist='StudentsT', rescale=False)
            res = am.fit(disp='off')
            cond_vol = res.conditional_volatility
            # predicted 1-step ahead sigma approximated by previous day's conditional volatility
            pred_sigma = cond_vol.shift(1).reindex(r.index)
            # extract nu if Student-t
            try:
                if hasattr(res, 'params') and 'nu' in res.params.index:
                    nu = float(res.params['nu'])
            except Exception:
                nu = None
            try:
                model_info['aic'] = float(res.aic)
                model_info['bic'] = float(res.bic)
            except Exception:
                pass
        except Exception as e:
            print('[WARN] GARCH fit failed:', e)
            pred_sigma = pd.Series(index=r.index, data=[np.nan]*len(r))
            nu = None

    df = pd.DataFrame({'date': r.index, 'realized_ret': r.values, 'pred_sigma': pred_sigma.values})
    df['realized_vol_abs'] = np.abs(df['realized_ret'])
    df['roll_realized_vol_21'] = df['realized_ret'].rolling(window=21).std()
    if nu is not None:
        df['nu'] = nu
    return df, model_info

def compute_var_es(df, alpha_list, assume='auto'):
    """Compute VaR and ES for each row using predicted sigma. Assume zero mean for simplicity.
    If assume=='t' supply df['nu'] column else normal.
    """
    out = df.copy()
    for alpha in alpha_list:
        var_col = f'VaR_{int(alpha*100)}pct'
        es_col = f'ES_{int(alpha*100)}pct'
        vals = []
        ess = []
        for i, row in out.iterrows():
            sigma = row.get('pred_sigma', np.nan)
            if np.isnan(sigma) or sigma == 0:
                vals.append(np.nan); ess.append(np.nan); continue
            if assume == 't' and 'nu' in row and not np.isnan(row['nu']):
                nu = float(row['nu'])
                t_q = student_t.ppf(alpha, df=nu)
                VaR = - sigma * t_q
                # ES formula for Student-t (lower tail)
                pdf_val = student_t.pdf(t_q, df=nu)
                ES = - sigma * ( (nu + t_q**2) / (nu - 1) ) * pdf_val / alpha
                vals.append(VaR); ess.append(ES)
            else:
                z = norm.ppf(alpha)
                VaR = - sigma * z
                ES = - sigma * norm.pdf(z) / alpha
                vals.append(VaR); ess.append(ES)
        out[var_col] = vals
        out[es_col] = ess
    return out

def asset_report(file):
    base = os.path.splitext(os.path.basename(file))[0]
    ticker = base.replace('_data','')
    df_raw = pd.read_csv(file)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    if 'date' in df_raw.columns:
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.set_index('date')
    close_col = None
    for c in ['close','adj close','adj_close','price','close_price']:
        if c in df_raw.columns:
            close_col = c; break
    if close_col is None:
        print(f'[WARN] {file} missing close column, skipping')
        return
    returns = df_raw[close_col].pct_change().dropna()
    lookback = args.lookback_days
    df_fore, model_info = fit_garch_series(returns, lookback)
    # compute VaR/ES under normal approx (or t if nu available and dist=='t' or auto)
    assume = 'normal'
    if args.dist == 't':
        assume = 't'
    elif args.dist == 'auto' and ('nu' in df_fore.columns and not df_fore['nu'].isnull().all()):
        assume = 't'
    df_ve = compute_var_es(df_fore, args.alpha_list, assume=assume)
    # save CSV
    out_csv = os.path.join(args.outdir, f'{ticker}_vol_forecast.csv')
    df_ve.to_csv(out_csv, index=False)

    # plot predicted sigma vs realized abs returns and rolling vol
    plt.figure(figsize=(10,6))
    plt.plot(df_ve['date'], df_ve['pred_sigma'], label='Predicted sigma (1-step)')
    plt.plot(df_ve['date'], df_ve['realized_vol_abs'], alpha=0.6, label='Realized |r|')
    plt.plot(df_ve['date'], df_ve['roll_realized_vol_21'], alpha=0.6, label='Rolling vol (21d)')
    # mark VaR thresholds (horizontal lines)
    for a in args.alpha_list:
        col = f'VaR_{int(a*100)}pct'
        if col in df_ve.columns:
            try:
                y = df_ve[col].mean()
                plt.axhline(y=y, color='r', linestyle='--', alpha=0.4)
                xpos = df_ve['date'].iloc[max(0, len(df_ve)-50)]
                plt.text(xpos, y, f'VaR {int(a*100)}% ({a:.2%})', color='r')
            except Exception:
                pass
    plt.legend(); plt.grid(True)
    plt.title(f'{ticker} Predicted vs Realized Volatility')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'{ticker}_vol_forecast.png'))
    plt.close()

    # Save VaR/ES CSV (subset columns)
    cols = ['date','realized_ret','pred_sigma','realized_vol_abs','roll_realized_vol_21']
    for a in args.alpha_list:
        cols += [f'VaR_{int(a*100)}pct', f'ES_{int(a*100)}pct']
    df_ve[cols].to_csv(os.path.join(args.outdir, f'{ticker}_VaR_ES.csv'), index=False)

    # Write markdown report
    md_path = os.path.join(args.outdir, f'volatility_risk_report_{ticker}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Volatility & Risk Report - ' + ticker + '\n\n')
        f.write(f'- Data file: {file}\n')
        f.write(f'- Sample period: {returns.index.min().date()} -> {returns.index.max().date()}\n')
        f.write(f'- GARCH lookback (days): {lookback}\n')
        if model_info and model_info.get('aic') is not None:
            f.write(f"- Model AIC: {model_info['aic']:.2f}, BIC: {model_info['bic']:.2f}\n")
        f.write('\n## Plots\n')
        f.write(f'![Vol forecast]({ticker}_vol_forecast.png)\n')
        f.write('\n## VaR / ES (examples)\n')
        for a in args.alpha_list:
            f.write(f'- VaR {a:.2%}: saved in {ticker}_VaR_ES.csv column VaR_{int(a*100)}pct\n')
    print(f'[INFO] Written reports for {ticker} -> {out_csv}, markdown -> {md_path}')

def main():
    for file in FILES:
        try:
            asset_report(file)
        except Exception as e:
            print(f'[ERROR] processing {file}:', e)

if __name__ == '__main__':
    main()
