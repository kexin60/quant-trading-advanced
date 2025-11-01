"""
Implied Volatility Surface Modeling

This script provides comprehensive implied volatility surface modeling capabilities:
- Option chain data retrieval using yfinance and IBKR API
- IV surface construction with strike × maturity dimensions
- Cubic spline and SABR model fitting for volatility smile interpolation
- Time-series analysis of volatility smile structure evolution
- Advanced risk modeling and financial mathematics applications

Key Features:
- Multi-source option data collection (yfinance, IBKR)
- Black-Scholes implied volatility calculation
- SABR model calibration for volatility smile modeling
- 3D volatility surface visualization
- Term structure and skew analysis
- Historical volatility smile tracking

Outputs:
- IV surface plots (3D and heatmaps)
- SABR calibration results
- Volatility term structure analysis
- Skew evolution tracking
- Risk metrics and Greeks surface

Usage:
python "Implied Volatility Surface Modeling.py" --symbol SPY --source yfinance
python "Implied Volatility Surface Modeling.py" --symbol AAPL --source ibkr --live
"""

import os
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import concurrent.futures
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from scipy import interpolate
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import scipy.sparse as sp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional dependencies with graceful fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARN] yfinance not available. Install with: pip install yfinance")

try:
    from ib_insync import IB, Stock, Option, util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("[WARN] ib_insync not available. Install with: pip install ib-insync")

# Configuration
REPORT_DIR = 'reports'
os.makedirs(REPORT_DIR, exist_ok=True)

# Black-Scholes Functions
def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def implied_volatility_newton(price, S, K, T, r, option_type='call', max_iter=100, tol=1e-6):
    """Calculate implied volatility using Newton-Raphson method"""
    if T <= 0:
        return np.nan
    
    # Initial guess
    sigma = 0.3
    
    for i in range(max_iter):
        if option_type.lower() == 'call':
            bs_price = black_scholes_call(S, K, T, r, sigma)
        else:
            bs_price = black_scholes_put(S, K, T, r, sigma)
        
        # Vega calculation
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        if abs(vega) < 1e-10:
            return np.nan
        
        # Newton-Raphson update
        diff = bs_price - price
        if abs(diff) < tol:
            return sigma
        
        sigma = sigma - diff / vega
        
        # Bounds check
        if sigma <= 0.001:
            sigma = 0.001
        elif sigma >= 5.0:
            sigma = 5.0
    
    return sigma if sigma > 0.001 and sigma < 5.0 else np.nan

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0 or sigma <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan, 'rho': np.nan}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # Per 1% change in vol
    theta = theta / 365  # Per day
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

# SABR Model Implementation
class SABRModel:
    """SABR (Stochastic Alpha Beta Rho) volatility model"""
    
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.nu = None
        self.rho = None
    
    def sabr_volatility(self, F, K, T, alpha, beta, nu, rho):
        """Calculate SABR implied volatility"""
        eps = 1e-7
        logFK = np.log(F / K)
        FK = (F * K) ** ((1 - beta) / 2)
        
        if abs(logFK) < eps:
            # ATM case
            numer1 = (((1 - beta) ** 2) / 24) * (alpha ** 2) / (FK ** 2)
            numer2 = 0.25 * rho * beta * nu * alpha / FK
            numer3 = ((2 - 3 * (rho ** 2)) / 24) * (nu ** 2)
            VolAtm = alpha * (1 + (numer1 + numer2 + numer3) * T) / FK
            return VolAtm
        else:
            # Non-ATM case
            z = (nu / alpha) * FK * logFK
            Xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            
            numer1 = (((1 - beta) ** 2) / 24) * ((alpha ** 2) / (FK ** 2))
            numer2 = 0.25 * rho * beta * nu * alpha / FK
            numer3 = ((2 - 3 * (rho ** 2)) / 24) * (nu ** 2)
            
            numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
            denom1 = FK * ((1 + (((1 - beta) ** 2) / 24) * (logFK ** 2) + 
                           (((1 - beta) ** 4) / 1920) * (logFK ** 4)))
            denom = denom1 * Xz
            
            return numer / denom
    
    def calibrate(self, strikes, maturities, market_vols, forward_price, bounds=None):
        """Calibrate SABR parameters to market volatilities"""
        if bounds is None:
            bounds = [(0.01, 2.0),    # alpha
                     (0.1, 0.99),    # beta  
                     (0.01, 2.0),    # nu
                     (-0.99, 0.99)]  # rho
        
        def objective(params):
            alpha, beta, nu, rho = params
            sse = 0.0
            count = 0
            
            for (K, T), market_vol in zip(zip(strikes, maturities), market_vols):
                if np.isfinite(market_vol) and market_vol > 0:
                    try:
                        model_vol = self.sabr_volatility(forward_price, K, T, alpha, beta, nu, rho)
                        if np.isfinite(model_vol) and model_vol > 0:
                            sse += (model_vol - market_vol) ** 2
                            count += 1
                    except:
                        continue
            
            return sse / max(count, 1)
        
        # Use differential evolution for global optimization
        result = differential_evolution(objective, bounds, maxiter=1000, seed=42)
        
        if result.success:
            self.alpha, self.beta, self.nu, self.rho = result.x
            return result.x, result.fun
        else:
            # Fallback to scipy.optimize.minimize
            x0 = [0.3, 0.5, 0.3, 0.0]  # Initial guess
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            self.alpha, self.beta, self.nu, self.rho = result.x
            return result.x, result.fun

# Data Retrieval Classes
class YFinanceDataProvider:
    """Option data provider using yfinance"""
    
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.ticker = None
        
    def get_option_chain(self, expiration_dates=None):
        """Retrieve option chain data"""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not available")
        
        try:
            self.ticker = yf.Ticker(self.symbol)
            
            # Get current stock price
            hist = self.ticker.history(period="1d")
            if hist.empty:
                raise ValueError(f"No price data available for {self.symbol}")
            current_price = hist['Close'].iloc[-1]
            
            # Get available expiration dates
            if expiration_dates is None:
                expiration_dates = self.ticker.options[:6]  # First 6 expirations
            
            option_data = []
            
            for exp_date in expiration_dates:
                try:
                    opt_chain = self.ticker.option_chain(exp_date)
                    
                    # Process calls
                    for _, row in opt_chain.calls.iterrows():
                        if row['bid'] > 0 and row['ask'] > 0 and row['volume'] > 0:
                            mid_price = (row['bid'] + row['ask']) / 2
                            option_data.append({
                                'expiration': exp_date,
                                'strike': row['strike'],
                                'option_type': 'call',
                                'price': mid_price,
                                'bid': row['bid'],
                                'ask': row['ask'],
                                'volume': row['volume'],
                                'openInterest': row['openInterest'],
                                'impliedVolatility': row['impliedVolatility']
                            })
                    
                    # Process puts
                    for _, row in opt_chain.puts.iterrows():
                        if row['bid'] > 0 and row['ask'] > 0 and row['volume'] > 0:
                            mid_price = (row['bid'] + row['ask']) / 2
                            option_data.append({
                                'expiration': exp_date,
                                'strike': row['strike'],
                                'option_type': 'put',
                                'price': mid_price,
                                'bid': row['bid'],
                                'ask': row['ask'],
                                'volume': row['volume'],
                                'openInterest': row['openInterest'],
                                'impliedVolatility': row['impliedVolatility']
                            })
                            
                except Exception as e:
                    print(f"[WARN] Failed to get options for {exp_date}: {e}")
                    continue
            
            df = pd.DataFrame(option_data)
            if df.empty:
                raise ValueError("No option data retrieved")
            
            return df, current_price
            
        except Exception as e:
            raise Exception(f"Failed to retrieve option data: {e}")

class IBKRDataProvider:
    """Option data provider using Interactive Brokers API"""
    
    def __init__(self, symbol, host='127.0.0.1', port=7497, client_id=2):
        self.symbol = symbol.upper()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        
    def connect(self):
        """Connect to IBKR"""
        if not IBKR_AVAILABLE:
            raise ImportError("ib_insync not available")
        
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            print(f"Connected to IBKR ({self.host}:{self.port})")
        except Exception as e:
            raise Exception(f"Failed to connect to IBKR: {e}")
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib:
            self.ib.disconnect()
    
    def get_option_chain(self, days_to_expiry=None):
        """Retrieve option chain data from IBKR"""
        if not self.ib:
            self.connect()
        
        try:
            # Get underlying stock contract
            stock = Stock(self.symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            # Get current price
            ticker = self.ib.reqMktData(stock, '', False, False)
            self.ib.sleep(2)  # Wait for price data
            current_price = ticker.last or ticker.close
            if not current_price:
                raise ValueError(f"Could not get current price for {self.symbol}")
            
            # Get option chains
            chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
            
            option_data = []
            for chain in chains[:3]:  # Limit to first 3 chains to avoid too much data
                for expiration in sorted(chain.expirations)[:6]:  # First 6 expirations
                    try:
                        exp_date = datetime.strptime(expiration, '%Y%m%d')
                        days_to_exp = (exp_date - datetime.now()).days
                        
                        if days_to_expiry and days_to_exp not in days_to_expiry:
                            continue
                        
                        # Get strikes around current price
                        strikes = sorted(chain.strikes)
                        atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
                        strike_range = strikes[max(0, atm_index-10):min(len(strikes), atm_index+11)]
                        
                        for strike in strike_range:
                            # Call option
                            call_contract = Option(self.symbol, expiration, strike, 'C', 'SMART')
                            put_contract = Option(self.symbol, expiration, strike, 'P', 'SMART')
                            
                            contracts = [call_contract, put_contract]
                            self.ib.qualifyContracts(*contracts)
                            
                            for contract in contracts:
                                try:
                                    ticker = self.ib.reqMktData(contract, '', False, False)
                                    self.ib.sleep(1)
                                    
                                    if ticker.bid and ticker.ask and ticker.bid > 0:
                                        mid_price = (ticker.bid + ticker.ask) / 2
                                        option_data.append({
                                            'expiration': expiration,
                                            'strike': strike,
                                            'option_type': 'call' if contract.right == 'C' else 'put',
                                            'price': mid_price,
                                            'bid': ticker.bid,
                                            'ask': ticker.ask,
                                            'volume': ticker.volume or 0,
                                            'openInterest': 0  # Not readily available in IBKR
                                        })
                                except Exception as e:
                                    continue
                                    
                    except Exception as e:
                        print(f"[WARN] Failed to process expiration {expiration}: {e}")
                        continue
            
            df = pd.DataFrame(option_data)
            if df.empty:
                raise ValueError("No option data retrieved from IBKR")
            
            return df, current_price
            
        except Exception as e:
            raise Exception(f"Failed to retrieve IBKR option data: {e}")

# Volatility Surface Builder
class VolatilitySurface:
    """Implied Volatility Surface constructor and analyzer"""
    
    def __init__(self, symbol, risk_free_rate=0.05):
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.option_data = None
        self.current_price = None
        self.iv_surface = None
        self.sabr_model = SABRModel()
        
    def load_data(self, source='yfinance', **kwargs):
        """Load option data from specified source"""
        if source.lower() == 'yfinance':
            provider = YFinanceDataProvider(self.symbol)
            self.option_data, self.current_price = provider.get_option_chain()
        elif source.lower() == 'ibkr':
            provider = IBKRDataProvider(self.symbol, **kwargs)
            try:
                self.option_data, self.current_price = provider.get_option_chain()
            finally:
                provider.disconnect()
        else:
            raise ValueError("Source must be 'yfinance' or 'ibkr'")
        
        print(f"Loaded {len(self.option_data)} option contracts for {self.symbol}")
        print(f"Current price: ${self.current_price:.2f}")
        
    def calculate_implied_volatilities(self):
        """Calculate implied volatilities for all options"""
        if self.option_data is None:
            raise ValueError("No option data loaded")
        
        iv_data = []
        
        for _, row in self.option_data.iterrows():
            # Calculate time to expiration
            if isinstance(row['expiration'], str):
                exp_date = datetime.strptime(row['expiration'], '%Y-%m-%d')
            else:
                exp_date = pd.to_datetime(row['expiration'])
            
            T = (exp_date - datetime.now()).days / 365.0
            
            if T <= 0:
                continue
            
            # Use market IV if available, otherwise calculate
            if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']) and row['impliedVolatility'] > 0:
                iv = row['impliedVolatility']
            else:
                # Calculate IV using Newton-Raphson
                iv = implied_volatility_newton(
                    row['price'], 
                    self.current_price, 
                    row['strike'], 
                    T, 
                    self.risk_free_rate, 
                    row['option_type']
                )
            
            if pd.notna(iv) and 0.01 <= iv <= 3.0:  # Reasonable IV bounds
                # Calculate moneyness
                moneyness = row['strike'] / self.current_price
                
                # Calculate Greeks
                greeks = calculate_greeks(
                    self.current_price, 
                    row['strike'], 
                    T, 
                    self.risk_free_rate, 
                    iv, 
                    row['option_type']
                )
                
                iv_data.append({
                    'expiration': row['expiration'],
                    'days_to_expiry': T * 365,
                    'strike': row['strike'],
                    'moneyness': moneyness,
                    'option_type': row['option_type'],
                    'market_price': row['price'],
                    'implied_vol': iv,
                    'volume': row.get('volume', 0),
                    'open_interest': row.get('openInterest', 0),
                    **greeks
                })
        
        self.iv_surface = pd.DataFrame(iv_data)
        print(f"Calculated IV for {len(self.iv_surface)} options")
        
        return self.iv_surface
    
    def fit_sabr_model(self, expiration_filter=None):
        """Fit SABR model to volatility smile"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        sabr_results = {}
        
        # Group by expiration
        exp_groups = self.iv_surface.groupby('expiration')
        
        for exp_date, group in exp_groups:
            if expiration_filter and exp_date not in expiration_filter:
                continue
                
            if len(group) < 5:  # Need minimum data points
                continue
            
            # Focus on liquid options
            liquid_group = group[
                (group['volume'] > 0) | (group['open_interest'] > 0)
            ].copy()
            
            if len(liquid_group) < 3:
                liquid_group = group.copy()
            
            # Prepare data for SABR calibration
            strikes = liquid_group['strike'].values
            market_vols = liquid_group['implied_vol'].values
            T = liquid_group['days_to_expiry'].iloc[0] / 365.0
            maturities = np.full_like(strikes, T)
            
            # Calibrate SABR
            try:
                sabr_model = SABRModel()
                params, rmse = sabr_model.calibrate(
                    strikes, maturities, market_vols, self.current_price
                )
                
                # Generate fitted volatilities
                fitted_vols = []
                for K in strikes:
                    fitted_vol = sabr_model.sabr_volatility(
                        self.current_price, K, T, *params
                    )
                    fitted_vols.append(fitted_vol)
                
                sabr_results[exp_date] = {
                    'params': params,
                    'rmse': rmse,
                    'strikes': strikes,
                    'market_vols': market_vols,
                    'fitted_vols': fitted_vols,
                    'days_to_expiry': T * 365
                }
                
                print(f"SABR fit for {exp_date}: α={params[0]:.3f}, β={params[1]:.3f}, "
                      f"ν={params[2]:.3f}, ρ={params[3]:.3f}, RMSE={rmse:.4f}")
                
            except Exception as e:
                print(f"[WARN] SABR calibration failed for {exp_date}: {e}")
                continue
        
        return sabr_results
    
    def create_volatility_surface_3d(self, method='cubic'):
        """Create 3D volatility surface using interpolation"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        # Prepare grid data
        strikes = self.iv_surface['strike'].values
        days_to_expiry = self.iv_surface['days_to_expiry'].values
        implied_vols = self.iv_surface['implied_vol'].values
        
        # Create regular grid
        strike_range = np.linspace(strikes.min(), strikes.max(), 50)
        days_range = np.linspace(days_to_expiry.min(), days_to_expiry.max(), 30)
        
        Strike_grid, Days_grid = np.meshgrid(strike_range, days_range)
        
        # Interpolate using specified method
        if method == 'cubic':
            # Cubic spline interpolation
            points = np.column_stack((strikes, days_to_expiry))
            Vol_grid = interpolate.griddata(
                points, implied_vols, (Strike_grid, Days_grid), method='cubic'
            )
            # Fill NaN values with linear interpolation
            Vol_grid_linear = interpolate.griddata(
                points, implied_vols, (Strike_grid, Days_grid), method='linear'
            )
            Vol_grid = np.where(np.isnan(Vol_grid), Vol_grid_linear, Vol_grid)
            
        elif method == 'rbf':
            # Radial basis function interpolation
            rbf = interpolate.Rbf(strikes, days_to_expiry, implied_vols, function='cubic')
            Vol_grid = rbf(Strike_grid, Days_grid)
        
        return Strike_grid, Days_grid, Vol_grid
    
    def analyze_volatility_smile(self):
        """Analyze volatility smile characteristics"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        smile_analysis = {}
        
        # Group by expiration
        exp_groups = self.iv_surface.groupby('expiration')
        
        for exp_date, group in exp_groups:
            if len(group) < 3:
                continue
            
            # Sort by moneyness
            group_sorted = group.sort_values('moneyness')
            
            # Find ATM vol (closest to moneyness = 1.0)
            atm_idx = (group_sorted['moneyness'] - 1.0).abs().idxmin()
            atm_vol = group_sorted.loc[atm_idx, 'implied_vol']
            
            # Calculate skew metrics
            otm_puts = group_sorted[group_sorted['moneyness'] < 0.95]
            otm_calls = group_sorted[group_sorted['moneyness'] > 1.05]
            
            if len(otm_puts) > 0 and len(otm_calls) > 0:
                put_vol = otm_puts['implied_vol'].mean()
                call_vol = otm_calls['implied_vol'].mean()
                skew = put_vol - call_vol
            else:
                skew = np.nan
            
            # Calculate term structure slope
            days_to_exp = group['days_to_expiry'].iloc[0]
            
            smile_analysis[exp_date] = {
                'days_to_expiry': days_to_exp,
                'atm_vol': atm_vol,
                'skew': skew,
                'vol_range': group['implied_vol'].max() - group['implied_vol'].min(),
                'num_options': len(group)
            }
        
        return pd.DataFrame.from_dict(smile_analysis, orient='index')
    
    def plot_volatility_surface(self, save_path=None):
        """Plot 3D volatility surface"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        # Create interpolated surface
        Strike_grid, Days_grid, Vol_grid = self.create_volatility_surface_3d()
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(Strike_grid, Days_grid, Vol_grid * 100,
                              cmap=cm.RdYlBu_r, alpha=0.8, linewidth=0, antialiased=True)
        
        # Scatter plot of actual data points
        ax.scatter(self.iv_surface['strike'], self.iv_surface['days_to_expiry'], 
                  self.iv_surface['implied_vol'] * 100, 
                  c='black', s=20, alpha=0.6)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Days to Expiry')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title(f'{self.symbol} Implied Volatility Surface')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(REPORT_DIR, f'{self.symbol}_iv_surface_3d.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        
    def plot_volatility_heatmap(self, save_path=None):
        """Plot 2D volatility heatmap"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        # Pivot data for heatmap
        pivot_data = self.iv_surface.pivot_table(
            values='implied_vol', 
            index='days_to_expiry', 
            columns='strike', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data * 100, cmap='RdYlBu_r', cbar_kws={'label': 'Implied Volatility (%)'})
        plt.title(f'{self.symbol} Implied Volatility Heatmap')
        plt.xlabel('Strike Price')
        plt.ylabel('Days to Expiry')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(REPORT_DIR, f'{self.symbol}_iv_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_volatility_smile(self, sabr_results=None, save_path=None):
        """Plot volatility smile for different expirations"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        exp_groups = self.iv_surface.groupby('expiration')
        
        for i, (exp_date, group) in enumerate(exp_groups):
            if i >= 4:  # Limit to 4 subplots
                break
            
            ax = axes[i]
            
            # Sort by moneyness for smooth line
            group_sorted = group.sort_values('moneyness')
            
            # Plot market data
            ax.plot(group_sorted['moneyness'], group_sorted['implied_vol'] * 100, 
                   'o-', label='Market', linewidth=2, markersize=6)
            
            # Plot SABR fit if available
            if sabr_results and exp_date in sabr_results:
                sabr_data = sabr_results[exp_date]
                strikes = sabr_data['strikes']
                fitted_vols = sabr_data['fitted_vols']
                moneyness = strikes / self.current_price
                
                # Sort for smooth line
                sort_idx = np.argsort(moneyness)
                ax.plot(moneyness[sort_idx], np.array(fitted_vols)[sort_idx] * 100, 
                       '--', label='SABR', linewidth=2)
            
            ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='ATM')
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(f'Expiration: {exp_date}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(i+1, 4):
            axes[j].set_visible(False)
        
        plt.suptitle(f'{self.symbol} Volatility Smile Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(REPORT_DIR, f'{self.symbol}_volatility_smile.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_term_structure(self, save_path=None):
        """Plot implied volatility term structure"""
        if self.iv_surface is None:
            raise ValueError("No IV surface calculated")
        
        # Calculate ATM volatility for each expiration
        term_structure_data = []
        
        exp_groups = self.iv_surface.groupby('expiration')
        
        for exp_date, group in exp_groups:
            # Find ATM option (closest to moneyness = 1.0)
            atm_idx = (group['moneyness'] - 1.0).abs().idxmin()
            atm_vol = group.loc[atm_idx, 'implied_vol']
            days_to_exp = group.loc[atm_idx, 'days_to_expiry']
            
            term_structure_data.append({
                'expiration': exp_date,
                'days_to_expiry': days_to_exp,
                'atm_vol': atm_vol
            })
        
        ts_df = pd.DataFrame(term_structure_data).sort_values('days_to_expiry')
        
        plt.figure(figsize=(10, 6))
        plt.plot(ts_df['days_to_expiry'], ts_df['atm_vol'] * 100, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Days to Expiry')
        plt.ylabel('ATM Implied Volatility (%)')
        plt.title(f'{self.symbol} Implied Volatility Term Structure')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(REPORT_DIR, f'{self.symbol}_term_structure.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return ts_df
    
    def generate_report(self, sabr_results=None):
        """Generate comprehensive volatility surface report"""
        report_path = os.path.join(REPORT_DIR, f'{self.symbol}_iv_surface_report.md')
        
        # Analyze volatility smile
        smile_analysis = self.analyze_volatility_smile()
        
        with open(report_path, 'w') as f:
            f.write(f'# {self.symbol} Implied Volatility Surface Analysis\n\n')
            f.write(f'**Analysis Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write(f'**Current Stock Price:** ${self.current_price:.2f}\n\n')
            f.write(f'**Risk-Free Rate:** {self.risk_free_rate:.2%}\n\n')
            
            # Summary statistics
            f.write('## Summary Statistics\n\n')
            f.write(f'- Total Option Contracts: {len(self.option_data)}\n')
            f.write(f'- Options with Valid IV: {len(self.iv_surface)}\n')
            f.write(f'- Average Implied Volatility: {self.iv_surface["implied_vol"].mean():.2%}\n')
            f.write(f'- IV Range: {self.iv_surface["implied_vol"].min():.2%} - {self.iv_surface["implied_vol"].max():.2%}\n\n')
            
            # Volatility smile analysis
            f.write('## Volatility Smile Analysis\n\n')
            f.write(smile_analysis.to_string())
            f.write('\n\n')
            
            # SABR model results
            if sabr_results:
                f.write('## SABR Model Calibration Results\n\n')
                f.write('| Expiration | Days to Expiry | α | β | ν | ρ | RMSE |\n')
                f.write('|------------|---------------|---|---|---|---|------|\n')
                
                for exp_date, result in sabr_results.items():
                    params = result['params']
                    f.write(f'| {exp_date} | {result["days_to_expiry"]:.0f} | '
                           f'{params[0]:.3f} | {params[1]:.3f} | {params[2]:.3f} | '
                           f'{params[3]:.3f} | {result["rmse"]:.4f} |\n')
                f.write('\n')
            
            # File outputs
            f.write('## Generated Files\n\n')
            f.write(f'- 3D Volatility Surface: `{self.symbol}_iv_surface_3d.png`\n')
            f.write(f'- Volatility Heatmap: `{self.symbol}_iv_heatmap.png`\n')
            f.write(f'- Volatility Smile: `{self.symbol}_volatility_smile.png`\n')
            f.write(f'- Term Structure: `{self.symbol}_term_structure.png`\n')
            f.write(f'- Raw Data: `{self.symbol}_iv_surface_data.csv`\n')
        
        # Save raw data
        self.iv_surface.to_csv(
            os.path.join(REPORT_DIR, f'{self.symbol}_iv_surface_data.csv'), 
            index=False
        )
        
        print(f"Report generated: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Implied Volatility Surface Modeling')
    parser.add_argument('--symbol', type=str, default='SPY', 
                       help='Stock symbol to analyze')
    parser.add_argument('--source', type=str, choices=['yfinance', 'ibkr'], 
                       default='yfinance', help='Data source')
    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                       help='Risk-free rate for calculations')
    parser.add_argument('--live', action='store_true',
                       help='Use live data (for IBKR)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='IBKR host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7497,
                       help='IBKR port (default: 7497)')
    
    args = parser.parse_args()
    
    try:
        # Initialize volatility surface
        vs = VolatilitySurface(args.symbol, args.risk_free_rate)
        
        # Load option data
        print(f"Loading option data for {args.symbol} from {args.source}...")
        if args.source == 'ibkr':
            vs.load_data(args.source, host=args.host, port=args.port)
        else:
            vs.load_data(args.source)
        
        # Calculate implied volatilities
        print("Calculating implied volatilities...")
        vs.calculate_implied_volatilities()
        
        # Fit SABR model
        print("Fitting SABR model...")
        sabr_results = vs.fit_sabr_model()
        
        # Generate visualizations
        print("Generating visualizations...")
        vs.plot_volatility_surface()
        vs.plot_volatility_heatmap()
        vs.plot_volatility_smile(sabr_results)
        vs.plot_term_structure()
        
        # Generate report
        print("Generating comprehensive report...")
        vs.generate_report(sabr_results)
        
        print(f"\nAnalysis complete! Check the '{REPORT_DIR}' directory for outputs.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()