# Multi-Asset Portfolio Optimization
- Mean-Variance efficient frontier (scipy)
- Approximate Max-Sharpe via risk-aversion sweep (quadratic program)
- CVaR minimization frontier (cvxpy) if cvxpy is installed
- Weight sensitivity analysis across risk tolerance (lambda sweep)

# Outputs:
- efficient_frontier_meanvar.png
- efficient_frontier_cvar.png (if cvxpy available)
- weight_paths.png
- portfolio_opt_report.txt

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Attempt to import cvxpy for CVaR optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

# CLI / Configuration
p = argparse.ArgumentParser()
p.add_argument('--rf', type=float, default=float(os.getenv('RISK_FREE_RATE', '0.02')))
p.add_argument('--alpha', type=float, default=float(os.getenv('CVaR_ALPHA', '0.95')))
p.add_argument('--target-points', type=int, default=int(os.getenv('TARGET_POINTS','40')))
p.add_argument('--outdir', type=str, default=os.getenv('REPORT_DIR','reports'))
p.add_argument('--files-glob', type=str, default=os.getenv('FILES_GLOB','*_data.csv'))
p.add_argument('--allow-short', action='store_true')
p.add_argument('--max-weight', type=float, default=float(os.getenv('MAX_WEIGHT', '0.4')))
p.add_argument('--max-short', type=float, default=float(os.getenv('MAX_SHORT', '0.4')))
p.add_argument('--cov-ridge-eps', type=float, default=float(os.getenv('COV_RIDGE_EPS','1e-8')))
p.add_argument('--l2', type=float, default=float(os.getenv('L2_REG', '0.0')),
               help='L2 regularization on weights (gamma)')
p.add_argument('--tc-l1', type=float, default=float(os.getenv('TC_L1', '0.0')),
               help='Transaction cost L1 penalty (not used in basic optimization)')
p.add_argument('--rolling', action='store_true', help='Enable simple monthly rolling reopt backtest')
p.add_argument('--lookback-days', type=int, default=int(os.getenv('LOOKBACK_DAYS','252')))
p.add_argument('--rebalance-days', type=int, default=int(os.getenv('REBALANCE_DAYS','21')),
               help='Rebalance every N trading days when --rolling is enabled')
args = p.parse_args()

# 自动发现工作目录下所有 *_data.csv 文件（更灵活）
FILES = sorted(glob.glob(args.files_glob))
if not FILES:
    # 兼容旧命名：尝试包含 _data 的 CSV 文件
    FILES = sorted([f for f in os.listdir('.') if '_data' in f and f.endswith('.csv')])
if not FILES:
    raise FileNotFoundError('未在当前目录找到匹配的数据文件，请检查 --files-glob 或生成数据。')
RISK_FREE_RATE = args.rf
TRADING_DAYS = 252
REPORT_DIR = args.outdir
os.makedirs(REPORT_DIR, exist_ok=True)

# 常用约束（CLI/环境变量覆盖）
MAX_WEIGHT = args.max_weight
ALLOW_SHORT = args.allow_short
MAX_SHORT = args.max_short
CVAR_ALPHA = args.alpha
COV_RIDGE_EPS = args.cov_ridge_eps
L2_REG = args.l2
TC_L1 = args.tc_l1
REBALANCE_DAYS = args.rebalance_days


# Load price series and compute returns
price_data = {}
for file in FILES:
    # 更健壮的 symbol 解析，支持带下划线的 ticker，例如 BRK_B_data.csv
    base = os.path.splitext(file)[0]        # e.g., "BRK_B_data"
    symbol = base.replace('_data', '')      # -> "BRK_B"
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    # try common close column names
    close_col = None
    for c in ['close','adj close','adj_close','price','close_price']:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        # 提示用户可用列名以便手动选择或修正 CSV
        cols = ', '.join(df.columns.tolist())
        raise ValueError(f'在 {file} 中未能识别收盘价列。可用列名: {cols}. 请确保存在 close/adj close/price 类似列。')
    price_data[symbol] = df[close_col]

prices = pd.concat(price_data, axis=1)
# 报告对齐前后的样本信息，提醒用户如果丢失太多样本
print(f"[INFO] 原始各资产样本长度: {[len(s) for s in price_data.values()]} ")
# 每个资产的有效起止日期，方便排查谁缺数据
for s, series in price_data.items():
    non_na = series.dropna()
    if len(non_na) > 0:
        try:
            print(f"[INFO] {s}: {non_na.index.min().date()} -> {non_na.index.max().date()}, N={len(non_na)}")
        except Exception:
            print(f"[INFO] {s}: N={len(non_na)} (dates unavailable)")
    else:
        print(f"[WARN] {s}: no valid data")
prices = prices.dropna()
returns = prices.pct_change().dropna()
print(f"[INFO] 对齐后时间区间: {prices.index.min().date()} -> {prices.index.max().date()}，样本数 T={len(returns)}")

# Statistics
mu = returns.mean() * TRADING_DAYS  # annualized
cov = returns.cov() * TRADING_DAYS  # annualized
# 数值稳健性：轻微岭化，防止病态协方差矩阵导致优化失败
# 使用 CLI/args 中的 COV_RIDGE_EPS，而不是再次从环境变量读取
eps = float(COV_RIDGE_EPS)
cov = pd.DataFrame(cov.values + np.eye(len(cov)) * eps, index=cov.index, columns=cov.columns)
assets = list(prices.columns)
R = returns.values  # T x n matrix of sample returns
T, n = R.shape

# Helper functions
def port_return(weights):
    return float(weights @ mu.values)

def port_vol(weights):
    return float(np.sqrt(weights.T @ cov.values @ weights))

# Mean-variance frontier (existing approach) -> returns list of (vol, ret)
def mean_variance_frontier(mu, cov, targets):
    n = len(mu)
    # 构造 bounds 支持空头选项
    if ALLOW_SHORT:
        lb = -MAX_SHORT
    else:
        lb = 0.0
    ub = MAX_WEIGHT
    bounds = [(lb, ub)] * n
    cons = ({'type':'eq','fun':lambda w: np.sum(w)-1},)
    vols = []
    weights_list = []
    for r in targets:
        c = (
            {'type':'eq','fun':lambda w: np.sum(w)-1},
            {'type':'eq','fun':(lambda w, mu=mu, r=r: float(np.dot(w, mu) - r))}
        )
        # 目标里加入 L2 正则以提高可交易性/稳定性
        def obj_mv(w, cov=cov, l2=L2_REG):
            return float(w @ cov @ w) + l2 * float(w @ w)
        res = minimize(obj_mv, np.ones(n)/n, bounds=bounds, constraints=c)
        if not res.success:
            vols.append(np.nan)
            weights_list.append(None)
        else:
            w = res.x
            # 真实波动率应基于协方差矩阵，而不是包含正则项的目标值
            vols.append(np.sqrt(float(w @ cov @ w)))
            weights_list.append(w)
    return np.array(vols), weights_list

# Approximate max-Sharpe by sweeping quadratic objective: maximize (mu-rf)^T w - (lambda/2) w^T Sigma w
def max_sharpe_via_risk_aversion(mu, cov, lambdas):
    n = len(mu)
    # 支持约束（多头/空头/单资产上限）
    if ALLOW_SHORT:
        lb = -MAX_SHORT
    else:
        lb = 0.0
    ub = MAX_WEIGHT
    bounds = [(lb, ub)] * n
    cons = ({'type':'eq','fun':lambda w: np.sum(w)-1},)
    best = {'sharpe': -1e9}
    records = []
    for lam in lambdas:
        def obj(w, mu=mu, cov=cov, lam=lam, l2=L2_REG):
            # maximize risk-adjusted return approx -> minimize negative utility
            return -(w @ (mu - RISK_FREE_RATE)) + (lam/2.0) * (w @ cov @ w) + l2 * (w @ w)
        res = minimize(obj, np.ones(n)/n, bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            r = w @ mu
            vol = np.sqrt(w @ cov @ w)
            sharpe = (r - RISK_FREE_RATE) / vol if vol>0 else 0.0
            records.append((lam, w, r, vol, sharpe))
            if sharpe > best['sharpe']:
                best = {'lam': lam, 'w': w, 'r': r, 'vol': vol, 'sharpe': sharpe}
    return best, records

# 风险调整收益指标：Sortino ratio 与最大回撤（基于日收益序列）
def sortino_ratio(returns_series, required_return=0.0, periods_per_year=TRADING_DAYS):
    # returns_series: daily returns (pandas Series or 1d array)
    r = np.asarray(returns_series)
    if len(r) == 0:
        return np.nan
    # 年化 excess return
    mean_excess = (np.nanmean(r) - required_return/periods_per_year) * periods_per_year
    # downside deviation
    downside = r[r < (required_return/periods_per_year)]
    dd = np.sqrt(np.nanmean(downside**2)) * np.sqrt(periods_per_year) if len(downside) > 0 else 0.0
    if dd == 0:
        return np.nan
    return mean_excess / dd

def max_drawdown(returns_series):
    r = np.asarray(returns_series)
    if len(r) == 0:
        return 0.0
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peak) / peak
    return float(np.min(drawdowns))

# CVaR minimization frontier using cvxpy
def cvar_minimization_frontier(R, target_returns, alpha=0.95):
    # R: T x n sample returns
    if not CVXPY_AVAILABLE:
        raise RuntimeError('cvxpy 未安装，无法计算 CVaR 前沿')
    T, n = R.shape
    cvar_values = []
    weights_list = []
    for r_target in target_returns:
        w = cp.Variable(n)
        z = cp.Variable()
        xi = cp.Variable(T)
        # portfolio returns for each sample: R @ w
        port_ret = R @ w  # shape T
        loss = -port_ret
        # CVaR objective: z + (1/(alpha*T)) sum xi
        obj = cp.Minimize(z + (1.0/(1-alpha)/T) * cp.sum(xi))
        # 与全局约束一致：根据 ALLOW_SHORT 设置上下界
        if ALLOW_SHORT:
            lb = -MAX_SHORT
        else:
            lb = 0.0
        ub = MAX_WEIGHT
        constraints = [xi >= 0,
                       xi >= loss - z,
                       cp.sum(w) == 1,
                       w >= lb,
                       w <= ub,
                       w @ mu.values >= r_target]
        # 如果需要 L2 正则，可在 CVX 问题中加入 gamma * norm(w,2)^2
        if L2_REG and L2_REG > 0:
            obj = cp.Minimize(z + (1.0/(1-alpha)/T) * cp.sum(xi) + L2_REG * cp.sum_squares(w))
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception as e:
                print('cvxpy solve error:', e)
                cvar_values.append(np.nan)
                weights_list.append(None)
                continue
        if w.value is None:
            cvar_values.append(np.nan)
            weights_list.append(None)
        else:
            # compute CVaR numeric
            wval = np.array(w.value).flatten()
            port_rets = R @ wval
            losses = -port_rets
            zval = float(z.value)
            xi_val = np.maximum(losses - zval, 0.0)
            cvar_est = zval + xi_val.mean()/(1-alpha)
            cvar_values.append(cvar_est)
            weights_list.append(wval)
    return np.array(cvar_values), weights_list

# Weight sensitivity: sweep risk aversion lambda and record weights
def weight_sensitivity(mu, cov, lambdas):
    weights = []
    for lam in lambdas:
        n = len(mu)
        if ALLOW_SHORT:
            lb = -MAX_SHORT
        else:
            lb = 0.0
        ub = MAX_WEIGHT
        bounds = [(lb, ub)] * n
        cons = ({'type':'eq','fun':lambda w: np.sum(w)-1},)
        def obj(w, mu=mu, cov=cov, lam=lam, l2=L2_REG):
            return (lam/2.0) * (w @ cov @ w) - (w @ (mu - RISK_FREE_RATE)) + l2 * (w @ w)
        res = minimize(obj, np.ones(n)/n, bounds=bounds, constraints=cons)
        if res.success:
            weights.append(res.x)
        else:
            weights.append(np.full(n, np.nan))
    return np.array(weights)


def optimize_for_lambda(mu, cov, lam):
    """直接对单个 lambda 求解得到权重（返回 None 或权重向量）。"""
    n = len(mu)
    if ALLOW_SHORT:
        lb = -MAX_SHORT
    else:
        lb = 0.0
    ub = MAX_WEIGHT
    bounds = [(lb, ub)] * n
    cons = ({'type':'eq','fun':lambda w: np.sum(w)-1},)
    def obj(w, mu=mu, cov=cov, lam=lam, l2=L2_REG):
        return -(w @ (mu - RISK_FREE_RATE)) + (lam/2.0) * (w @ cov @ w) + l2 * (w @ w)
    res = minimize(obj, np.ones(n)/n, bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    return None


def rolling_backtest(R_all, dates_index, lookback_days=252, rebalance_days=21, lam=None):
    """Simple rolling rebalancing backtest.
    - Re-estimate mu/cov on lookback window and optimize for given lam.
    - Rebalance every rebalance_days and apply weights to next period returns.
    Returns: daily DataFrame with columns: date, daily_ret, cum_ret
    """
    T, n = R_all.shape
    if T <= lookback_days:
        raise ValueError('样本不足，无法运行 rolling backtest（T <= lookback_days）')
    # portfolio returns placeholder
    daily_rets = np.zeros(T)
    weights_ts = []
    # start at first day where we have lookback available
    t = lookback_days
    while t < T:
        ins_start = t - lookback_days
        ins_end = t
        R_ins = R_all[ins_start:ins_end, :]
        mu_ins = R_ins.mean(axis=0) * TRADING_DAYS
        cov_ins = np.cov(R_ins.T) * TRADING_DAYS
        cov_ins = cov_ins + np.eye(n) * COV_RIDGE_EPS
        # choose weights via optimize_for_lambda; fall back to equal weight if fail
        w = None
        if lam is not None:
            w = optimize_for_lambda(mu_ins, cov_ins, lam)
        if w is None:
            # fallback: try a short lambda grid to find feasible w
            w = optimize_for_lambda(mu_ins, cov_ins, lam=1.0)
        if w is None:
            w = np.ones(n)/n
        weights_ts.append((t, w))
        out_end = min(t + rebalance_days, T)
        for tt in range(t, out_end):
            daily_rets[tt] = float(R_all[tt] @ w)
        t = out_end

    # Build DataFrame for out-of-sample period
    dates_oos = dates_index
    df = pd.DataFrame({'date': dates_oos, 'daily_ret': daily_rets})
    # crop to after lookback (where we started producing returns)
    df = df.iloc[lookback_days:].reset_index(drop=True)
    df['cum_ret'] = np.cumprod(1 + df['daily_ret']) - 1.0
    # compute rolling metrics
    ann_ret = ((1 + df['cum_ret'].iloc[-1]) ** (TRADING_DAYS / len(df))) - 1 if len(df) > 0 else np.nan
    ann_vol = df['daily_ret'].std() * np.sqrt(TRADING_DAYS) if len(df) > 1 else np.nan
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    sortino = sortino_ratio(df['daily_ret'].values) if len(df) > 0 else np.nan
    mdd = max_drawdown(df['daily_ret'].values) if len(df) > 0 else np.nan
    metrics = {'ann_return': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': mdd}
    return df, metrics, weights_ts

# Main
if __name__ == '__main__':
    # 可行性检查：若禁止做空且 MAX_WEIGHT 过小，自动放宽并提示
    if not ALLOW_SHORT and MAX_WEIGHT < 1.0 / n:
        print(f"[WARN] MAX_WEIGHT={MAX_WEIGHT} 过小，n={n} 时至少需要 >= {1.0/n:.3f} 才可行。已自动提升。")
        MAX_WEIGHT = 1.0 / n

    # Mean-variance frontier：使用更稳健的 target_returns（基于分位数）
    low_q = float(os.getenv('TARGET_LOW_Q', '0.10'))
    high_q = float(os.getenv('TARGET_HIGH_Q', '0.90'))
    low = np.quantile(mu.values, low_q)
    high = np.quantile(mu.values, high_q)
    target_returns = np.linspace(low, high, args.target_points)
    vols_mv, weights_mv = mean_variance_frontier(mu.values, cov.values, target_returns)

    # Max Sharpe via lambda sweep（自适应分辨率）
    base = 60
    resolution = min(300, max(base, int(base * (n / 3))))
    lam_grid = np.logspace(-3, 3, resolution)
    best_sharpe, records = max_sharpe_via_risk_aversion(mu.values, cov.values, lam_grid)

    # 失败保护：确保找到了有效的解
    if not np.isfinite(best_sharpe.get('sharpe', -np.inf)) or best_sharpe.get('sharpe', -1e99) < -1e8:
        raise RuntimeError('未找到可行的最大夏普解，请检查约束/数据窗口/参数（例如 MAX_WEIGHT 太小 或 数据太短）。')

    # Weight sensitivity（自适应分辨率）
    lambdas = np.logspace(-3, 3, max(50, int(50 * (n / 3))))
    w_paths = weight_sensitivity(mu.values, cov.values, lambdas)

    # CVaR frontier (if available)
    cvar_vals = None
    weights_cvar = None
    if CVXPY_AVAILABLE:
        try:
            cvar_vals, weights_cvar = cvar_minimization_frontier(R, target_returns, alpha=args.alpha)
        except Exception as e:
            print('计算 CVaR 前沿失败:', e)
            CVXPY_AVAILABLE = False

    # Save plots
    # Mean-variance frontier
    plt.figure(figsize=(8,5))
    plt.plot(vols_mv, target_returns, 'b--', label='Mean-Variance Frontier')
    plt.scatter(best_sharpe['vol'], best_sharpe['r'], c='r', marker='*', s=150, label='Approx Max Sharpe')
    # 标注最优点
    try:
        plt.annotate('Max Sharpe', xy=(best_sharpe['vol'], best_sharpe['r']), xytext=(best_sharpe['vol']*1.05, best_sharpe['r']*0.98),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red')
    except Exception:
        pass
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier (Mean-Variance)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    mv_path = os.path.join(REPORT_DIR, 'efficient_frontier_meanvar.png')
    plt.savefig(mv_path)
    plt.close()

    # 导出 Mean-Variance 前沿数据为 CSV（包含权重）
    mv_rows = []
    for r, v, w in zip(target_returns, vols_mv, weights_mv):
        row = {'target_return': float(r), 'volatility': float(v) if not np.isnan(v) else np.nan}
        if w is not None:
            for i, a in enumerate(assets):
                row[f'w_{a}'] = float(w[i])
        else:
            for a in assets:
                row[f'w_{a}'] = np.nan
        mv_rows.append(row)
    mv_df = pd.DataFrame(mv_rows)
    mv_csv = os.path.join(REPORT_DIR, 'mv_frontier.csv')
    mv_df.to_csv(mv_csv, index=False)

    # CVaR frontier plot (CVaR on x-axis)
    if CVXPY_AVAILABLE and cvar_vals is not None:
        plt.figure(figsize=(8,5))
        plt.plot(cvar_vals, target_returns, 'g--', label=f'CVaR Frontier ({args.alpha:.2f})')
        plt.xlabel('CVaR (loss)')
        plt.ylabel('Expected Return')
        plt.title(f'Efficient Frontier (CVaR minimization, alpha={args.alpha:.2f})')
        plt.legend(); plt.grid(True); plt.tight_layout()
        cvar_path = os.path.join(REPORT_DIR, 'efficient_frontier_cvar.png')
        plt.savefig(cvar_path)
        plt.close()
        # 导出 CVaR 前沿为 CSV
        cvar_rows = []
        for r, c, w in zip(target_returns, cvar_vals, weights_cvar):
            row = {'target_return': float(r), 'cvar': float(c) if not np.isnan(c) else np.nan}
            if w is not None:
                for i, a in enumerate(assets):
                    row[f'w_{a}'] = float(w[i])
            else:
                for a in assets:
                    row[f'w_{a}'] = np.nan
            cvar_rows.append(row)
        cvar_df = pd.DataFrame(cvar_rows)
        cvar_csv = os.path.join(REPORT_DIR, 'cvar_frontier.csv')
        cvar_df.to_csv(cvar_csv, index=False)

    # Weight paths plot
    plt.figure(figsize=(10,6))
    for i, a in enumerate(assets):
        plt.plot(lambdas, w_paths[:, i], label=a)
    # 在权重路径图上标注用于近似 Max-Sharpe 的 lambda（如果在网格中）
    try:
        lam_best = best_sharpe['lam']
        plt.axvline(lam_best, color='k', linestyle=':', alpha=0.6)
        plt.text(lam_best, 0.02, 'lambda*', rotation=90, verticalalignment='bottom')
    except Exception:
        pass
    plt.xscale('log')
    plt.xlabel('Risk aversion (lambda)')
    plt.ylabel('Asset weight')
    plt.title('Weight sensitivity vs risk aversion')
    plt.legend(); plt.grid(True); plt.tight_layout()
    wp_path = os.path.join(REPORT_DIR, 'weight_paths.png')
    plt.savefig(wp_path)
    plt.close()

    # 导出 weight paths 为 CSV
    wp_df = pd.DataFrame(w_paths, columns=assets)
    wp_df.insert(0, 'lambda', lambdas)
    wp_csv = os.path.join(REPORT_DIR, 'weight_paths.csv')
    wp_df.to_csv(wp_csv, index=False)

    # Write report
    report_lines = []
    report_lines.append('Multi-Asset Portfolio Optimization Report\n')
    report_lines.append('Assets: ' + ', '.join(assets) + '\n')
    report_lines.append('\n-- Mean-Variance Optimal Weights (approx) --\n')
    report_lines.append('Max-Sharpe approx (lambda={:.6f})\n'.format(best_sharpe['lam']))
    for a, w in zip(assets, best_sharpe['w']):
        report_lines.append(f'{a}: {w:.4f}\n')
    report_lines.append(f'Expected Return: {best_sharpe["r"]:.4f}, Vol: {best_sharpe["vol"]:.4f}, Sharpe: {best_sharpe["sharpe"]:.4f}\n')

    # 计算并输出风险调整收益指标（基于样本日收益）
    # 用最佳权重估计组合的日收益序列
    try:
        w_best = best_sharpe['w']
        port_daily_rets = R @ w_best
        sortino = sortino_ratio(port_daily_rets)
        mdd = max_drawdown(port_daily_rets)
        report_lines.append(f'Sortino Ratio (annualized): {sortino:.4f}\n')
        report_lines.append(f'Max Drawdown: {mdd:.2%}\n')
    except Exception:
        report_lines.append('无法计算 Sortino/MaxDrawdown（样本或权重问题）\n')

    # 保存最佳组合为 JSON，便于复现/回测
    try:
        import json
        best_path = os.path.join(REPORT_DIR, 'best_portfolio.json')
        json.dump({
            'lambda': float(best_sharpe['lam']),
            'weights': {a: float(w) for a, w in zip(assets, best_sharpe['w'])},
            'expected_return': float(best_sharpe['r']),
            'vol': float(best_sharpe['vol']),
            'sharpe': float(best_sharpe['sharpe']),
            'sortino': float(sortino) if 'sortino' in locals() and sortino is not None else None,
            'max_drawdown': float(mdd) if 'mdd' in locals() else None
        }, open(best_path, 'w'), indent=2)
    except Exception:
        pass

    if CVXPY_AVAILABLE and cvar_vals is not None:
        report_lines.append('\n-- CVaR Frontier (95%) --\n')
        report_lines.append('TargetReturn, CVaR\n')
        for r, c in zip(target_returns, cvar_vals):
            report_lines.append(f'{r:.6f}, {c:.6f}\n')
    else:
        report_lines.append('\nCVaR analysis skipped (cvxpy not available)\n')

    report_lines.append('\nPlots saved to reports/\n')
    with open(os.path.join(REPORT_DIR, 'portfolio_opt_report.txt'),'w',encoding='utf-8') as f:
        f.writelines(report_lines)

    # If tc-l1 was provided, warn that it's currently a placeholder
    if TC_L1 and TC_L1 != 0.0:
        print(f"[WARN] --tc-l1 ({TC_L1}) currently not implemented in optimization and will be ignored.")

    # 生成 Markdown 汇总，嵌入图片与 CSV 链接
    md_lines = []
    md_lines.append('# Multi-Asset Portfolio Optimization Report\n\n')
    md_lines.append('## Summary\n')
    md_lines.append('Assets: ' + ', '.join(assets) + '\n\n')
    md_lines.append('### Max-Sharpe (approx)\n')
    md_lines.append(f'- lambda: {best_sharpe["lam"]:.6g}\n')
    md_lines.append(f'- Sharpe: {best_sharpe["sharpe"]:.4f}\n')
    md_lines.append('\n|Asset|Weight|\n|---:|---:|\n')
    for a, w in zip(assets, best_sharpe['w']):
        md_lines.append(f'|{a}|{w:.4f}|\n')
    md_lines.append('\n## Plots\n')
    md_lines.append(f'![Mean-Variance Frontier]({os.path.basename(mv_path)})\n')
    if CVXPY_AVAILABLE and cvar_vals is not None:
        md_lines.append(f'![CVaR Frontier]({os.path.basename(cvar_path)})\n')
    md_lines.append(f'![Weight paths]({os.path.basename(wp_path)})\n')
    md_lines.append('\n## CSV outputs\n')
    md_lines.append(f'- MV frontier: {os.path.basename(mv_csv)}\n')
    if CVXPY_AVAILABLE and cvar_vals is not None:
        md_lines.append(f'- CVaR frontier: {os.path.basename(cvar_csv)}\n')
    md_lines.append(f'- Weight paths: {os.path.basename(wp_csv)}\n')
    md_path = os.path.join(REPORT_DIR, 'portfolio_opt_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

    # If rolling requested, run the simple rolling backtest and append outputs
    if args.rolling:
        try:
            print(f"[INFO] Running rolling backtest ...")
            df_roll, metrics_roll, weights_ts = rolling_backtest(
                R, returns.index,
                lookback_days=args.lookback_days,
                rebalance_days=args.rebalance_days,
                lam=best_sharpe['lam']
            )
            df_roll.to_csv(os.path.join(REPORT_DIR, 'rolling_backtest.csv'), index=False)
            plt.figure(figsize=(8,5))
            plt.plot(df_roll['date'], df_roll['cum_ret'])
            plt.grid(True)
            plt.title('Rolling Backtest Cumulative Return')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.tight_layout()
            plt.savefig(os.path.join(REPORT_DIR, 'rolling_cumret.png'))
            plt.close()
            # append rolling summary to markdown
            with open(md_path, 'a', encoding='utf-8') as f:
                f.write('\n## Rolling Backtest\n')
                f.write(f"- Lookback days: {args.lookback_days}\n- Rebalance days: {args.rebalance_days}\n")
                f.write('- Files: rolling_backtest.csv, rolling_cumret.png\n')
        except Exception as e:
            print('[WARN] rolling backtest failed:', e)

    # Save parameters snapshot to JSON and include in report
    try:
        params = {
            'rf': RISK_FREE_RATE,
            'alpha': CVAR_ALPHA,
            'allow_short': ALLOW_SHORT,
            'max_weight': MAX_WEIGHT,
            'max_short': MAX_SHORT,
            'l2': L2_REG,
            'cov_ridge_eps': COV_RIDGE_EPS,
            'files_glob': args.files_glob,
            'target_points': args.target_points,
            'target_low_q': low_q,
            'target_high_q': high_q,
            'lookback_days': args.lookback_days,
            'rebalance_days': args.rebalance_days
        }
        # merge into existing best_portfolio.json if exists
        try:
            bp = json.load(open(best_path, 'r'))
        except Exception:
            bp = {}
        bp['params'] = params
        json.dump(bp, open(best_path, 'w'), indent=2)
        # append params to markdown
        with open(md_path, 'a', encoding='utf-8') as f:
            f.write('\n## Parameters\n')
            for k, v in params.items():
                f.write(f'- {k}: {v}\n')
    except Exception:
        pass

    print('Finished. Reports and plots saved under reports/')

    if not CVXPY_AVAILABLE:
        print('Note: cvxpy not detected — CVaR frontier was skipped. To enable CVaR add cvxpy: pip install cvxpy')
