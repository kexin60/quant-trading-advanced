import os
import sys
import json
import time
import logging
import warnings
import hashlib
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Core scientific computing
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("WARNING: joblib not available. Parallel processing will be disabled.")

# Import dependencies with fallbacks
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    ARCH_AVAILABLE = False
    print(f"WARNING: ARCH library not available ({e}). GARCH modeling will be disabled.")

try:
    from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, Order
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("WARNING: ib_insync not available. IBKR integration will be disabled.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not available. Yahoo Finance data source will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: Plotting libraries not available. Visualization will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configuration Management

@dataclass
class TradingConfig:
    """Configuration class for trading engine parameters"""
    
    # Trading Parameters
    symbols: List[str] = None
    initial_capital: float = 100000.0
    max_position_pct: float = 0.2  # Maximum position size as percentage of portfolio
    max_leverage: float = 1.0
    min_trade_interval: int = 60  # Minimum seconds between trades for same symbol
    
    # Risk Management
    max_drawdown: float = 0.15  # Maximum allowable drawdown
    var_confidence: float = 0.05  # VaR confidence level (5% = 95% VaR)
    position_limit: float = 0.25  # Maximum position weight per asset
    stop_loss_pct: float = 0.05  # Stop loss percentage
    take_profit_pct: float = 0.10  # Take profit percentage
    
    # Signal Generation
    lookback_days: int = 252  # Trading days for historical analysis
    signal_threshold_buy: float = 0.65
    signal_threshold_sell: float = 0.35
    ensemble_weights: Dict[str, float] = None  # Weights for signal fusion
    
    # Model Parameters
    ml_train_size: float = 0.7
    ml_cv_folds: int = 5
    ml_random_state: int = 42
    feature_selection_k: int = 10
    
    # Market Data
    refresh_interval: int = 5  # Seconds between market data updates
    data_sources: List[str] = None  # ['yahoo', 'ibkr', 'csv']
    
    # IBKR Configuration
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # Paper trading port
    ib_client_id: int = 1
    market_data_type: int = 3  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
    
    # Execution
    simulate_trading: bool = True
    order_type: str = "MARKET"  # MARKET, LIMIT, ADAPTIVE
    execution_algo: str = "ADAPTIVE"  # Order execution algorithm
    
    # Transaction Costs (for simulation)
    commission_rate: float = 0.005  # 0.5% commission rate
    bid_ask_spread: float = 0.001   # 0.1% bid-ask spread
    market_impact: float = 0.001    # 0.1% market impact cost
    
    # Portfolio-level Risk Controls
    portfolio_stop_loss_pct: float = 0.15      # 15% portfolio stop loss
    portfolio_take_profit_pct: float = 0.30    # 30% portfolio take profit
    max_daily_loss_pct: float = 0.05           # 5% max daily loss
    
    # Performance Optimization
    enable_feature_caching: bool = True         # Cache technical indicators
    enable_parallel_training: bool = True       # Parallel ML model training
    max_workers: int = 4                        # Number of parallel workers
    
    # Portfolio Rebalancing Control
    smooth_rebalancing: bool = True             # Enable smooth portfolio transitions
    max_turnover_per_day: float = 0.20          # Maximum daily turnover (20% of portfolio)
    rebalancing_steps: int = 5                  # Number of steps to reach target weights
    min_weight_change: float = 0.02             # Minimum weight change to trigger rebalancing
    transaction_cost_threshold: float = 0.01    # Skip trades if cost exceeds this threshold
    
    # Reporting
    report_dir: str = "reports"
    log_level: str = "INFO"
    generate_plots: bool = True
    
    def __post_init__(self):
        """Initialize default values and validate configuration"""
        if self.symbols is None:
            self.symbols = ['QQQ', 'VOO', 'AAPL']
        
        if self.data_sources is None:
            self.data_sources = ['yahoo', 'csv']
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'technical': 0.3,
                'ml_signals': 0.4,
                'volatility': 0.2,
                'sentiment': 0.1
            }
        
        # Create report directory
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert 0 < self.max_position_pct <= 1.0, "max_position_pct must be between 0 and 1"
        assert 0 < self.var_confidence < 1.0, "var_confidence must be between 0 and 1"
        assert self.initial_capital > 0, "initial_capital must be positive"
        assert len(self.symbols) > 0, "At least one symbol must be specified"
        
        # Normalize ensemble weights
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            self.ensemble_weights = {k: v/total_weight for k, v in self.ensemble_weights.items()}
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TradingConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

# Performance and Risk Metrics

class PerformanceMetrics:
    """Calculate comprehensive performance and risk metrics"""
    
    @staticmethod
    def calculate_returns_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, 
                                rf_rate: float = 0.02) -> Dict[str, float]:
        """Calculate comprehensive return-based performance metrics"""
        if returns.empty:
            return {}
        
        returns = returns.dropna()
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (ann_return - rf_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        metrics = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
        }
        
        # Benchmark comparison metrics
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_returns = benchmark_returns.reindex(returns.index).dropna()
            if len(benchmark_returns) > 0:
                beta = np.cov(returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
                alpha = ann_return - (rf_rate + beta * ((1 + benchmark_returns.mean()) ** 252 - 1 - rf_rate))
                tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
                information_ratio = alpha / tracking_error if tracking_error > 0 else 0
                
                metrics.update({
                    'alpha': alpha,
                    'beta': beta,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio
                })
        
        return metrics
    
    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        if returns.empty:
            return np.nan, np.nan
        
        var = returns.quantile(confidence)
        cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
        
        return var, cvar

# Data Handler Module

class DataHandler:
    """Enhanced data acquisition and management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_cache = {}
        self.price_data = {}
        self.real_time_data = {}
        self.data_quality_scores = {}
        
        # Initialize logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_historical_data(self, symbols: List[str] = None, 
                           start_date: str = None, 
                           end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Load historical data from multiple sources with fallback"""
        if symbols is None:
            symbols = self.config.symbols
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=self.config.lookback_days * 2)).strftime('%Y-%m-%d')
        
        historical_data = {}
        
        for symbol in symbols:
            self.logger.info(f"Loading historical data for {symbol}")
            
            # Try each data source in order
            data = None
            for source in self.config.data_sources:
                try:
                    if source == 'yahoo' and YFINANCE_AVAILABLE:
                        data = self._load_yahoo_data(symbol, start_date, end_date)
                    elif source == 'csv':
                        data = self._load_csv_data(symbol)
                    elif source == 'ibkr' and IBKR_AVAILABLE:
                        data = self._load_ibkr_historical(symbol, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load {symbol} from {source}: {e}")
                    continue
            
            if data is not None and not data.empty:
                # Data quality assessment
                quality_score = self._assess_data_quality(data)
                self.data_quality_scores[symbol] = quality_score
                
                # Store processed data
                historical_data[symbol] = self._process_raw_data(data)
                self.price_data[symbol] = historical_data[symbol]
                
                self.logger.info(f"Loaded {len(data)} records for {symbol}, quality score: {quality_score:.2f}")
            else:
                self.logger.error(f"Failed to load data for {symbol} from any source")
        
        return historical_data
    
    def _load_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {symbol}")
        
        # Standardize column names
        data = data.reset_index()
        data = data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        return data
    
    def _load_csv_data(self, symbol: str) -> pd.DataFrame:
        """Load data from local CSV files"""
        # Look for CSV files with symbol name
        csv_files = [f for f in os.listdir('.') if f.startswith(symbol) and f.endswith('.csv')]
        
        if not csv_files:
            csv_files = [f for f in os.listdir('data') if f.startswith(symbol) and f.endswith('.csv')]
            csv_files = [os.path.join('data', f) for f in csv_files]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found for {symbol}")
        
        # Prefer files with 'data_yf' or 'data' in name
        data_file = None
        for pattern in ['data_yf', 'data']:
            matching = [f for f in csv_files if pattern in f]
            if matching:
                data_file = sorted(matching)[0]
                break
        
        if data_file is None:
            data_file = sorted(csv_files)[0]
        
        data = pd.read_csv(data_file)
        
        # Standardize column names (case-insensitive matching)
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                column_mapping[col] = 'date'
            elif 'open' in col_lower:
                column_mapping[col] = 'open'
            elif 'high' in col_lower:
                column_mapping[col] = 'high'
            elif 'low' in col_lower:
                column_mapping[col] = 'low'
            elif 'close' in col_lower:
                column_mapping[col] = 'close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'volume'
        
        data = data.rename(columns=column_mapping)
        
        if 'close' not in data.columns:
            raise ValueError(f"No close price column found in {data_file}")
        
        return data
    
    def _load_ibkr_historical(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data from IBKR (placeholder for future implementation)"""
        # This would connect to IBKR and fetch historical data
        # For now, return empty DataFrame
        self.logger.warning("IBKR historical data loading not implemented yet")
        return pd.DataFrame()
    
    def _process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean raw price data"""
        data = data.copy()
        
        # Ensure date column exists and is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        else:
            # If no date column, use index
            data['date'] = pd.to_datetime(data.index)
        
        # Ensure required columns exist
        required_cols = ['close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Debug data structure
        self.logger.debug(f"Data shape: {data.shape}")
        self.logger.debug(f"Data columns: {list(data.columns)}")
        self.logger.debug(f"Data dtypes: {data.dtypes}")
        
        # Remove duplicate columns if any
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Convert price columns to numeric
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in data.columns:
                try:
                    # Ensure we're working with a Series, not a DataFrame slice
                    column_data = data[col]
                    if isinstance(column_data, pd.DataFrame):
                        column_data = column_data.iloc[:, 0]  # Take first column if DataFrame
                    data[col] = pd.to_numeric(column_data, errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Error converting column {col} to numeric: {e}")
                    # Fallback: convert via string
                    try:
                        data[col] = data[col].astype(str).str.replace(',', '').astype(float)
                    except:
                        self.logger.error(f"Failed to convert column {col}, skipping")
        
        # Remove rows with invalid data
        data = data.dropna(subset=['date', 'close'])
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Calculate returns
        data['return'] = data['close'].pct_change()
        
        # Calculate technical indicators
        data = self._add_technical_indicators(data)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        data = data.copy()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        
        # Exponential moving averages
        for span in [12, 26]:
            data[f'ema_{span}'] = data['close'].ewm(span=span).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volatility measures
        data['volatility_10'] = data['return'].rolling(window=10).std()
        data['volatility_20'] = data['return'].rolling(window=20).std()
        
        # Volume indicators (if volume data available)
        if 'volume' in data.columns and not data['volume'].isna().all():
            data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma_10']
        
        return data
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality and return score (0-1)"""
        if data.empty:
            return 0.0
        
        score = 1.0
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_pct * 0.5
        
        # Check for outliers in returns
        if 'return' in data.columns:
            returns = data['return'].dropna()
            if len(returns) > 0:
                outlier_pct = ((np.abs(returns) > 3 * returns.std()).sum()) / len(returns)
                score -= outlier_pct * 0.3
        
        # Check for data gaps
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            date_diffs = dates.diff().dropna()
            if len(date_diffs) > 0:
                expected_freq = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else pd.Timedelta(days=1)
                large_gaps = (date_diffs > expected_freq * 2).sum()
                gap_penalty = min(large_gaps / len(date_diffs), 0.2)
                score -= gap_penalty
        
        return max(score, 0.0)
    
    def get_real_time_data(self, symbols: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Get real-time market data (placeholder for IBKR integration)"""
        if symbols is None:
            symbols = self.config.symbols
        
        # For now, return last available prices from historical data
        real_time_prices = {}
        for symbol in symbols:
            if symbol in self.price_data and not self.price_data[symbol].empty:
                last_row = self.price_data[symbol].iloc[-1]
                real_time_prices[symbol] = {
                    'price': float(last_row['close']),
                    'bid': float(last_row['close']) * 0.999,  # Simulated bid
                    'ask': float(last_row['close']) * 1.001,  # Simulated ask
                    'volume': float(last_row.get('volume', 0)),
                    'timestamp': datetime.now()
                }
        
        return real_time_prices

# Signal Generation Module

class SignalGenerator:
    """Advanced signal generation with multiple strategies and fusion"""
    
    def __init__(self, config: TradingConfig, data_handler: DataHandler):
        self.config = config
        self.data_handler = data_handler
        self.models = {}
        self.feature_scalers = {}
        self.signal_history = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
        # Feature caching
        self.feature_cache = {}
        self.cache_dir = os.path.join(config.report_dir, 'cache')
        if config.enable_feature_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize feature engineering parameters
        self.feature_windows = [5, 10, 20, 50]
        self.momentum_windows = [1, 5, 10, 20]
        
    def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train machine learning models for each symbol"""
        self.logger.info("Training ML models for signal generation")
        
        if self.config.enable_parallel_training and JOBLIB_AVAILABLE and len(data) > 1:
            return self._train_models_parallel(data)
        else:
            return self._train_models_sequential(data)
    
    def _train_models_parallel(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train models in parallel using joblib"""
        self.logger.info(f"Training models in parallel using {self.config.max_workers} workers")
        
        def train_single_symbol(symbol_data):
            symbol, df = symbol_data
            return symbol, self._train_single_symbol_model(symbol, df)
        
        # Parallel execution
        results = Parallel(n_jobs=self.config.max_workers, verbose=1)(
            delayed(train_single_symbol)((symbol, df)) for symbol, df in data.items()
        )
        
        # Collect results
        model_results = {}
        for symbol, result in results:
            if result is not None:
                model_results[symbol] = result
                self.models[symbol] = result
        
        self.logger.info(f"Trained models for {len(model_results)} symbols in parallel")
        return model_results
    
    def _train_models_sequential(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train models sequentially (original method)"""
        model_results = {}
        
        for symbol, df in data.items():
            result = self._train_single_symbol_model(symbol, df)
            if result is not None:
                model_results[symbol] = result
                self.models[symbol] = result
        
        return model_results
    
    def _train_single_symbol_model(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Train models for a single symbol"""
        self.logger.info(f"Training models for {symbol}")
        
        # Prepare features and targets
        features_df = self._engineer_features(df, symbol)
        if features_df.empty:
            self.logger.warning(f"No features available for {symbol}")
            return None
        
        # Create target variable (next period return classification)
        target = self._create_target_variable(df)
        
        # Align features and target
        common_index = features_df.index.intersection(target.index)
        if len(common_index) < 50:
            self.logger.warning(f"Insufficient data for {symbol}: {len(common_index)} samples")
            return None
            
        X = features_df.loc[common_index]
        y = target.loc[common_index]
        
        # Remove any remaining NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 30:
            self.logger.warning(f"Too few valid samples for {symbol}: {len(X)}")
            return None
        
        # Time series split for training/testing
        n_splits = min(self.config.ml_cv_folds, len(X) // 20)
        if n_splits < 2:
            self.logger.warning(f"Insufficient data for cross-validation for {symbol}")
            return None
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Train multiple models
        models_to_try = {
            'logistic': LogisticRegression(random_state=self.config.ml_random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=self.config.ml_random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=self.config.ml_random_state
            ),
            'ridge': RidgeClassifier(random_state=self.config.ml_random_state)
        }
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for model_name, model in models_to_try.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                self.logger.info(f"{symbol} - {model_name}: CV Score = {mean_score:.4f} (+/- {std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = model_name
                    
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name} for {symbol}: {e}")
        
        if best_model is None:
            self.logger.warning(f"No model could be trained for {symbol}")
            return None
        
        # Train best model on full dataset
        try:
            best_model.fit(X_scaled, y)
            self.logger.info(f"Best model for {symbol}: {best_model_name} (score: {best_score:.4f})")
            
            # Store model and associated data
            return {
                'model': best_model,
                'scaler': scaler,
                'features': list(X.columns),
                'cv_score': best_score,
                'model_type': best_model_name,
                'training_samples': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train best model for {symbol}: {e}")
            return None
    
    def _engineer_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Engineer features for ML models with caching"""
        
        # Generate cache key based on data hash
        if self.config.enable_feature_caching and symbol:
            data_hash = hashlib.md5(str(data.values).encode()).hexdigest()
            cache_key = f"{symbol}_{data_hash}_{len(data)}"
            cache_file = os.path.join(self.cache_dir, f"features_{cache_key}.pkl")
            
            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        features = pickle.load(f)
                    self.logger.debug(f"Loaded cached features for {symbol}")
                    return features
                except Exception as e:
                    self.logger.warning(f"Failed to load cached features for {symbol}: {e}")
        
        # Compute features if not cached
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        if 'close' in df.columns:
            # Returns over different periods
            for window in self.momentum_windows:
                features[f'return_{window}d'] = df['close'].pct_change(window)
            
            # Moving average ratios
            for window in self.feature_windows:
                ma = df['close'].rolling(window=window).mean()
                features[f'price_ma_ratio_{window}'] = df['close'] / ma
            
            # Price percentiles over different windows
            for window in [20, 50, 100]:
                if len(df) > window:
                    features[f'price_percentile_{window}'] = df['close'].rolling(window=window).rank(pct=True)
        
        # Technical indicator features
        tech_indicators = ['rsi', 'macd', 'macd_signal', 'bb_position', 'bb_width']
        for indicator in tech_indicators:
            if indicator in df.columns:
                features[indicator] = df[indicator]
                # Add momentum of indicators
                features[f'{indicator}_momentum'] = df[indicator].diff()
        
        # Volatility features
        if 'return' in df.columns:
            for window in [5, 10, 20]:
                features[f'volatility_{window}'] = df['return'].rolling(window=window).std()
                features[f'volatility_rank_{window}'] = features[f'volatility_{window}'].rolling(window=window*2).rank(pct=True)
        
        # Volume features (if available)
        if 'volume' in df.columns and not df['volume'].isna().all():
            for window in [5, 10, 20]:
                vol_ma = df['volume'].rolling(window=window).mean()
                features[f'volume_ratio_{window}'] = df['volume'] / vol_ma
        
        # Cross-sectional features (relative strength)
        if 'return' in df.columns:
            for window in [5, 10, 20]:
                features[f'return_rank_{window}'] = df['return'].rolling(window=window).rank(pct=True)
        
        # Interaction features
        if 'rsi' in features.columns and 'bb_position' in features.columns:
            features['rsi_bb_interaction'] = features['rsi'] * features['bb_position']
        
        features_clean = features.dropna()
        
        # Save to cache if enabled
        if self.config.enable_feature_caching and symbol:
            try:
                cache_key = f"{symbol}_{data_hash}_{len(data)}"
                cache_file = os.path.join(self.cache_dir, f"features_{cache_key}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(features_clean, f)
                self.logger.debug(f"Cached features for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to cache features for {symbol}: {e}")
        
        return features_clean
    
    def _create_target_variable(self, data: pd.DataFrame, method: str = 'classification') -> pd.Series:
        """Create target variable for prediction"""
        if 'return' not in data.columns:
            data['return'] = data['close'].pct_change()
        
        if method == 'classification':
            # Binary classification: positive return vs negative return
            # Use a small buffer to avoid noise around zero
            buffer = 0.001  # 0.1% buffer
            target = (data['return'].shift(-1) > buffer).astype(int)
        elif method == 'regression':
            # Direct return prediction
            target = data['return'].shift(-1)
        else:
            raise ValueError(f"Unknown target method: {method}")
        
        return target
    
    def generate_ml_signals(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate ML-based trading signals"""
        signals = {}
        
        for symbol, df in current_data.items():
            if symbol not in self.models:
                self.logger.warning(f"No trained model available for {symbol}")
                continue
            
            try:
                # Prepare current features
                features_df = self._engineer_features(df, symbol)
                if features_df.empty:
                    continue
                
                # Get the most recent feature vector
                latest_features = features_df.iloc[-1]
                model_features = self.models[symbol]['features']
                
                # Ensure all required features are available
                if not all(feat in latest_features.index for feat in model_features):
                    missing_features = [f for f in model_features if f not in latest_features.index]
                    self.logger.warning(f"Missing features for {symbol}: {missing_features}")
                    continue
                
                X = latest_features[model_features].values.reshape(1, -1)
                
                # Check for NaN values
                if np.isnan(X).any():
                    self.logger.warning(f"NaN values in features for {symbol}")
                    continue
                
                # Scale features
                scaler = self.feature_scalers[symbol]
                X_scaled = scaler.transform(X)
                
                # Generate prediction
                model = self.models[symbol]['model']
                
                if hasattr(model, 'predict_proba'):
                    # Get probability of positive class
                    prob = model.predict_proba(X_scaled)[0, 1]
                    signals[symbol] = prob
                else:
                    # For models without probability prediction
                    prediction = model.predict(X_scaled)[0]
                    signals[symbol] = 0.75 if prediction == 1 else 0.25
                
                self.logger.debug(f"ML signal for {symbol}: {signals[symbol]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error generating ML signal for {symbol}: {e}")
                continue
        
        return signals
    
    def generate_technical_signals(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate technical analysis signals"""
        signals = {}
        
        for symbol, df in current_data.items():
            if df.empty or len(df) < 20:
                continue
            
            try:
                latest = df.iloc[-1]
                signal_components = []
                
                # RSI signal
                if 'rsi' in latest:
                    rsi = latest['rsi']
                    if rsi < 30:
                        rsi_signal = 0.8  # Oversold
                    elif rsi > 70:
                        rsi_signal = 0.2  # Overbought
                    else:
                        rsi_signal = 0.5
                    signal_components.append(rsi_signal)
                
                # MACD signal
                if 'macd' in latest and 'macd_signal' in latest:
                    macd_signal = 0.6 if latest['macd'] > latest['macd_signal'] else 0.4
                    signal_components.append(macd_signal)
                
                # Bollinger Bands signal
                if 'bb_position' in latest:
                    bb_pos = latest['bb_position']
                    if bb_pos < 0.2:
                        bb_signal = 0.7  # Near lower band
                    elif bb_pos > 0.8:
                        bb_signal = 0.3  # Near upper band
                    else:
                        bb_signal = 0.5
                    signal_components.append(bb_signal)
                
                # Moving average signal
                if 'ma_20' in latest and 'close' in latest:
                    ma_signal = 0.6 if latest['close'] > latest['ma_20'] else 0.4
                    signal_components.append(ma_signal)
                
                # Combine signals
                if signal_components:
                    signals[symbol] = np.mean(signal_components)
                else:
                    signals[symbol] = 0.5  # Neutral
                
                self.logger.debug(f"Technical signal for {symbol}: {signals[symbol]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error generating technical signal for {symbol}: {e}")
                signals[symbol] = 0.5
        
        return signals
    
    def generate_volatility_signals(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate volatility-based signals"""
        signals = {}
        
        for symbol, df in current_data.items():
            if df.empty or len(df) < 20:
                continue
            
            try:
                # Calculate recent volatility metrics
                returns = df['return'].dropna()
                if len(returns) < 10:
                    continue
                
                recent_vol = returns.tail(10).std()
                long_term_vol = returns.tail(50).std() if len(returns) >= 50 else returns.std()
                
                # Volatility regime signal
                vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1.0
                
                # Lower volatility might indicate trend continuation
                # Higher volatility might indicate mean reversion opportunity
                if vol_ratio < 0.8:
                    vol_signal = 0.6  # Low vol, trend continuation
                elif vol_ratio > 1.5:
                    vol_signal = 0.4  # High vol, mean reversion
                else:
                    vol_signal = 0.5  # Normal vol
                
                signals[symbol] = vol_signal
                self.logger.debug(f"Volatility signal for {symbol}: {signals[symbol]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error generating volatility signal for {symbol}: {e}")
                signals[symbol] = 0.5
        
        return signals
    
    def fuse_signals(self, ml_signals: Dict[str, float], 
                    technical_signals: Dict[str, float],
                    volatility_signals: Dict[str, float],
                    sentiment_signals: Dict[str, float] = None) -> Dict[str, float]:
        """Fuse multiple signal sources using ensemble weights"""
        
        all_symbols = set(ml_signals.keys()) | set(technical_signals.keys()) | set(volatility_signals.keys())
        if sentiment_signals:
            all_symbols |= set(sentiment_signals.keys())
        
        fused_signals = {}
        
        for symbol in all_symbols:
            signal_values = []
            weights = []
            
            # ML signals
            if symbol in ml_signals:
                signal_values.append(ml_signals[symbol])
                weights.append(self.config.ensemble_weights.get('ml_signals', 0.4))
            
            # Technical signals
            if symbol in technical_signals:
                signal_values.append(technical_signals[symbol])
                weights.append(self.config.ensemble_weights.get('technical', 0.3))
            
            # Volatility signals
            if symbol in volatility_signals:
                signal_values.append(volatility_signals[symbol])
                weights.append(self.config.ensemble_weights.get('volatility', 0.2))
            
            # Sentiment signals
            if sentiment_signals and symbol in sentiment_signals:
                signal_values.append(sentiment_signals[symbol])
                weights.append(self.config.ensemble_weights.get('sentiment', 0.1))
            
            if signal_values:
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Weighted average
                fused_signal = np.average(signal_values, weights=weights)
                fused_signals[symbol] = fused_signal
                
                # Store signal history for analysis
                self.signal_history[symbol].append({
                    'timestamp': datetime.now(),
                    'ml_signal': ml_signals.get(symbol),
                    'technical_signal': technical_signals.get(symbol),
                    'volatility_signal': volatility_signals.get(symbol),
                    'sentiment_signal': sentiment_signals.get(symbol) if sentiment_signals else None,
                    'fused_signal': fused_signal
                })
                
                # Keep only recent history
                if len(self.signal_history[symbol]) > 1000:
                    self.signal_history[symbol] = self.signal_history[symbol][-1000:]
        
        return fused_signals
    
    def get_signal_analysis(self, symbol: str, lookback_periods: int = 50) -> Dict[str, Any]:
        """Analyze signal performance and characteristics"""
        if symbol not in self.signal_history or len(self.signal_history[symbol]) < 2:
            return {}
        
        history = self.signal_history[symbol][-lookback_periods:]
        
        # Extract signal components
        timestamps = [h['timestamp'] for h in history]
        fused_signals = [h['fused_signal'] for h in history]
        ml_signals = [h['ml_signal'] for h in history if h['ml_signal'] is not None]
        technical_signals = [h['technical_signal'] for h in history if h['technical_signal'] is not None]
        
        analysis = {
            'signal_count': len(history),
            'avg_signal': np.mean(fused_signals),
            'signal_std': np.std(fused_signals),
            'signal_trend': np.polyfit(range(len(fused_signals)), fused_signals, 1)[0] if len(fused_signals) > 1 else 0
        }
        
        # Component analysis
        if ml_signals:
            analysis['ml_avg'] = np.mean(ml_signals)
            analysis['ml_std'] = np.std(ml_signals)
        
        if technical_signals:
            analysis['technical_avg'] = np.mean(technical_signals)
            analysis['technical_std'] = np.std(technical_signals)
        
        # Signal regime analysis
        strong_buy_pct = sum(1 for s in fused_signals if s > 0.7) / len(fused_signals)
        strong_sell_pct = sum(1 for s in fused_signals if s < 0.3) / len(fused_signals)
        
        analysis.update({
            'strong_buy_pct': strong_buy_pct,
            'strong_sell_pct': strong_sell_pct,
            'neutral_pct': 1 - strong_buy_pct - strong_sell_pct
        })
        
        return analysis

# Risk Management Module

class RiskManager:
    """Advanced risk management with VaR, position sizing, and regime detection"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.position_history = defaultdict(list)
        self.portfolio_history = []
        self.risk_metrics_history = []
        self.regime_detector = MarketRegimeDetector()
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.max_portfolio_var = 0.05  # Maximum daily VaR as % of portfolio
        self.max_concentration = 0.3   # Maximum single position weight
        self.max_sector_exposure = 0.5 # Maximum sector exposure
        
    def calculate_position_sizes(self, signals: Dict[str, float], 
                                current_prices: Dict[str, float],
                                portfolio_value: float,
                                current_positions: Dict[str, int] = None) -> Dict[str, int]:
        """Calculate optimal position sizes based on signals and risk constraints"""
        
        if current_positions is None:
            current_positions = {}
        
        # Calculate target weights from signals
        target_weights = self._signals_to_weights(signals)
        
        # Apply risk constraints
        target_weights = self._apply_risk_constraints(target_weights, current_prices, portfolio_value)
        
        # Convert weights to share quantities
        position_sizes = {}
        for symbol, weight in target_weights.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                target_value = weight * portfolio_value
                shares = int(target_value / current_prices[symbol])
                
                # Apply position limits
                max_shares = int(self.config.max_position_pct * portfolio_value / current_prices[symbol])
                shares = min(shares, max_shares)
                
                position_sizes[symbol] = shares
            else:
                position_sizes[symbol] = 0
        
        return position_sizes
    
    def _signals_to_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Convert trading signals to portfolio weights"""
        weights = {}
        
        # Transform signals to weights using a scaling function
        for symbol, signal in signals.items():
            # Convert signal (0-1) to weight (-max_weight to +max_weight)
            if signal > self.config.signal_threshold_buy:
                # Strong buy signal
                weight = min((signal - 0.5) * 2 * self.config.position_limit, self.config.position_limit)
            elif signal < self.config.signal_threshold_sell:
                # Strong sell signal (or short if allowed)
                if not self.config.simulate_trading:  # Only allow shorts in live trading
                    weight = max((signal - 0.5) * 2 * self.config.position_limit, -self.config.position_limit)
                else:
                    weight = 0  # No shorts in simulation
            else:
                # Neutral signal
                weight = 0
            
            weights[symbol] = weight
        
        # Normalize weights if total exceeds 1.0
        total_long_weight = sum(w for w in weights.values() if w > 0)
        total_short_weight = abs(sum(w for w in weights.values() if w < 0))
        
        if total_long_weight > 1.0:
            scale_factor = 1.0 / total_long_weight
            for symbol in weights:
                if weights[symbol] > 0:
                    weights[symbol] *= scale_factor
        
        return weights
    
    def _apply_risk_constraints(self, weights: Dict[str, float], 
                               prices: Dict[str, float], 
                               portfolio_value: float) -> Dict[str, float]:
        """Apply risk constraints to target weights"""
        
        # Check individual position limits
        for symbol in weights:
            weights[symbol] = max(min(weights[symbol], self.config.position_limit), 
                                -self.config.position_limit if not self.config.simulate_trading else 0)
        
        # Check portfolio concentration
        max_weight = max(abs(w) for w in weights.values()) if weights else 0
        if max_weight > self.max_concentration:
            scale_factor = self.max_concentration / max_weight
            weights = {k: v * scale_factor for k, v in weights.items()}
        
        return weights
    
    def calculate_portfolio_var(self, returns_data: Dict[str, pd.Series], 
                               weights: Dict[str, float], 
                               confidence: float = None) -> float:
        """Calculate portfolio Value at Risk"""
        
        if confidence is None:
            confidence = self.config.var_confidence
        
        # Align return series
        symbols = list(weights.keys())
        returns_df = pd.DataFrame({s: returns_data[s] for s in symbols if s in returns_data})
        returns_df = returns_df.dropna()
        
        if returns_df.empty or len(returns_df) < 30:
            return np.nan
        
        # Calculate portfolio returns
        weight_array = np.array([weights.get(s, 0) for s in returns_df.columns])
        portfolio_returns = (returns_df * weight_array).sum(axis=1)
        
        # Calculate VaR
        var = portfolio_returns.quantile(confidence)
        return var
    
    def calculate_expected_shortfall(self, returns_data: Dict[str, pd.Series], 
                                   weights: Dict[str, float], 
                                   confidence: float = None) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        
        if confidence is None:
            confidence = self.config.var_confidence
        
        # Calculate VaR first
        var = self.calculate_portfolio_var(returns_data, weights, confidence)
        
        if np.isnan(var):
            return np.nan
        
        # Align return series
        symbols = list(weights.keys())
        returns_df = pd.DataFrame({s: returns_data[s] for s in symbols if s in returns_data})
        returns_df = returns_df.dropna()
        
        # Calculate portfolio returns
        weight_array = np.array([weights.get(s, 0) for s in returns_df.columns])
        portfolio_returns = (returns_df * weight_array).sum(axis=1)
        
        # Calculate Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= var]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return expected_shortfall
    
    def check_risk_limits(self, current_positions: Dict[str, int], 
                         current_prices: Dict[str, float],
                         returns_data: Dict[str, pd.Series]) -> Dict[str, bool]:
        """Check if current portfolio violates risk limits"""
        
        risk_checks = {
            'position_limit_ok': True,
            'concentration_ok': True,
            'var_limit_ok': True,
            'drawdown_ok': True
        }
        
        if not current_positions or not current_prices:
            return risk_checks
        
        # Calculate current portfolio value and weights
        portfolio_value = sum(pos * current_prices.get(symbol, 0) 
                            for symbol, pos in current_positions.items())
        
        if portfolio_value <= 0:
            return risk_checks
        
        current_weights = {symbol: (pos * current_prices.get(symbol, 0)) / portfolio_value 
                          for symbol, pos in current_positions.items()}
        
        # Check position limits
        max_position_weight = max(abs(w) for w in current_weights.values()) if current_weights else 0
        risk_checks['position_limit_ok'] = max_position_weight <= self.config.position_limit
        
        # Check concentration
        risk_checks['concentration_ok'] = max_position_weight <= self.max_concentration
        
        # Check VaR
        portfolio_var = self.calculate_portfolio_var(returns_data, current_weights)
        if not np.isnan(portfolio_var):
            risk_checks['var_limit_ok'] = abs(portfolio_var) <= self.max_portfolio_var
        
        # Check drawdown (if we have portfolio history)
        if len(self.portfolio_history) > 0:
            current_value = portfolio_value
            peak_value = max(self.portfolio_history)
            current_drawdown = (current_value - peak_value) / peak_value if peak_value > 0 else 0
            risk_checks['drawdown_ok'] = current_drawdown >= -self.config.max_drawdown
        
        return risk_checks
    
    def get_risk_adjusted_signal(self, signal: float, symbol: str, 
                               current_positions: Dict[str, int],
                               current_prices: Dict[str, float]) -> float:
        """Adjust signal based on current risk exposure"""
        
        # Get current position weight
        portfolio_value = sum(pos * current_prices.get(sym, 0) 
                            for sym, pos in current_positions.items())
        
        if portfolio_value <= 0:
            return signal
        
        current_weight = (current_positions.get(symbol, 0) * current_prices.get(symbol, 0)) / portfolio_value
        
        # Reduce signal strength if position is already large
        if abs(current_weight) > self.config.position_limit * 0.8:
            signal_adjustment = 1 - (abs(current_weight) / self.config.position_limit)
            signal = 0.5 + (signal - 0.5) * signal_adjustment
        
        return signal
    
    def update_risk_metrics(self, portfolio_value: float, positions: Dict[str, int], 
                           prices: Dict[str, float], returns_data: Dict[str, pd.Series]):
        """Update risk metrics history"""
        
        # Calculate current weights
        weights = {}
        if portfolio_value > 0:
            weights = {symbol: (pos * prices.get(symbol, 0)) / portfolio_value 
                      for symbol, pos in positions.items()}
        
        # Calculate risk metrics
        var = self.calculate_portfolio_var(returns_data, weights)
        expected_shortfall = self.calculate_expected_shortfall(returns_data, weights)
        
        # Calculate diversification ratio
        portfolio_vol = self._calculate_portfolio_volatility(returns_data, weights)
        weighted_avg_vol = self._calculate_weighted_average_volatility(returns_data, weights)
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        risk_metrics = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'var_95': var,
            'expected_shortfall': expected_shortfall,
            'diversification_ratio': diversification_ratio,
            'max_position_weight': max(abs(w) for w in weights.values()) if weights else 0,
            'total_exposure': sum(abs(w) for w in weights.values())
        }
        
        self.risk_metrics_history.append(risk_metrics)
        self.portfolio_history.append(portfolio_value)
        
        # Keep only recent history
        if len(self.risk_metrics_history) > 1000:
            self.risk_metrics_history = self.risk_metrics_history[-1000:]
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def _calculate_portfolio_volatility(self, returns_data: Dict[str, pd.Series], 
                                      weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility"""
        symbols = list(weights.keys())
        returns_df = pd.DataFrame({s: returns_data[s] for s in symbols if s in returns_data})
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return 0.0
        
        weight_array = np.array([weights.get(s, 0) for s in returns_df.columns])
        portfolio_returns = (returns_df * weight_array).sum(axis=1)
        
        return portfolio_returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_weighted_average_volatility(self, returns_data: Dict[str, pd.Series], 
                                             weights: Dict[str, float]) -> float:
        """Calculate weighted average volatility of individual assets"""
        total_vol = 0.0
        total_weight = 0.0
        
        for symbol, weight in weights.items():
            if symbol in returns_data and abs(weight) > 0:
                asset_vol = returns_data[symbol].std() * np.sqrt(252)
                total_vol += abs(weight) * asset_vol
                total_weight += abs(weight)
        
        return total_vol / total_weight if total_weight > 0 else 0.0
    
    def check_portfolio_stop_loss_take_profit(self, current_portfolio_value: float, 
                                            initial_portfolio_value: float = None) -> Dict[str, Any]:
        """Check portfolio-level stop loss and take profit conditions"""
        
        if initial_portfolio_value is None:
            # Use the first recorded portfolio value as initial
            initial_portfolio_value = self.portfolio_history[0] if self.portfolio_history else current_portfolio_value
        
        if initial_portfolio_value <= 0:
            return {'action': 'hold', 'reason': 'invalid_initial_value'}
        
        # Calculate portfolio return
        portfolio_return = (current_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        # Check stop loss
        if portfolio_return <= -self.config.portfolio_stop_loss_pct:
            return {
                'action': 'stop_loss',
                'reason': f'Portfolio loss {portfolio_return:.2%} exceeds stop loss limit {self.config.portfolio_stop_loss_pct:.2%}',
                'portfolio_return': portfolio_return,
                'current_value': current_portfolio_value,
                'initial_value': initial_portfolio_value
            }
        
        # Check take profit
        if portfolio_return >= self.config.portfolio_take_profit_pct:
            return {
                'action': 'take_profit',
                'reason': f'Portfolio gain {portfolio_return:.2%} exceeds take profit limit {self.config.portfolio_take_profit_pct:.2%}',
                'portfolio_return': portfolio_return,
                'current_value': current_portfolio_value,
                'initial_value': initial_portfolio_value
            }
        
        # Check daily loss limit (if we have today's starting value)
        if len(self.portfolio_history) > 0:
            today_start_value = self.portfolio_history[-1]  # Assume last value is today's start
            daily_return = (current_portfolio_value - today_start_value) / today_start_value if today_start_value > 0 else 0
            
            if daily_return <= -self.config.max_daily_loss_pct:
                return {
                    'action': 'daily_stop_loss',
                    'reason': f'Daily loss {daily_return:.2%} exceeds daily loss limit {self.config.max_daily_loss_pct:.2%}',
                    'daily_return': daily_return,
                    'current_value': current_portfolio_value,
                    'today_start_value': today_start_value
                }
        
        return {
            'action': 'hold',
            'reason': 'within_risk_limits',
            'portfolio_return': portfolio_return,
            'current_value': current_portfolio_value,
            'initial_value': initial_portfolio_value
        }
    
    def apply_portfolio_risk_controls(self, signals: Dict[str, float], 
                                    current_portfolio_value: float,
                                    initial_portfolio_value: float = None) -> Dict[str, float]:
        """Apply portfolio-level risk controls to trading signals"""
        
        risk_check = self.check_portfolio_stop_loss_take_profit(current_portfolio_value, initial_portfolio_value)
        
        if risk_check['action'] in ['stop_loss', 'take_profit', 'daily_stop_loss']:
            self.logger.warning(f"Portfolio risk control triggered: {risk_check['action']}")
            self.logger.warning(f"Reason: {risk_check['reason']}")
            
            # Return zero signals to close all positions
            return {symbol: 0.0 for symbol in signals.keys()}
        
        # If within limits, return original signals
        return signals
    
    def calculate_smooth_rebalancing(self, current_weights: Dict[str, float], 
                                   target_weights: Dict[str, float],
                                   portfolio_value: float,
                                   current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate smooth rebalancing weights to avoid market impact"""
        
        if not self.config.smooth_rebalancing:
            return target_weights
        
        # Calculate weight differences
        weight_changes = {}
        total_turnover = 0.0
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            change = target_w - current_w
            weight_changes[symbol] = change
            total_turnover += abs(change)
        
        self.logger.info(f"Portfolio turnover needed: {total_turnover:.2%}")
        
        # If turnover is within daily limit, proceed with full rebalancing
        if total_turnover <= self.config.max_turnover_per_day:
            return target_weights
        
        # Otherwise, calculate gradual adjustment
        adjustment_factor = self.config.max_turnover_per_day / total_turnover
        adjusted_weights = {}
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            change = target_w - current_w
            
            # Apply gradual adjustment
            adjusted_change = change * adjustment_factor
            
            # Skip tiny changes to reduce transaction costs
            if abs(adjusted_change) < self.config.min_weight_change:
                adjusted_weights[symbol] = current_w
            else:
                adjusted_weights[symbol] = current_w + adjusted_change
        
        # Normalize weights to sum to 1.0
        total_weight = sum(abs(w) for w in adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {s: w / total_weight for s, w in adjusted_weights.items()}
        
        # Log the adjustment
        actual_turnover = sum(abs(adjusted_weights.get(s, 0) - current_weights.get(s, 0)) 
                             for s in all_symbols)
        self.logger.info(f"Adjusted turnover: {actual_turnover:.2%} (factor: {adjustment_factor:.2f})")
        
        return adjusted_weights
    
    def estimate_transaction_costs(self, weight_changes: Dict[str, float],
                                 portfolio_value: float,
                                 current_prices: Dict[str, float]) -> Dict[str, float]:
        """Estimate transaction costs for portfolio changes"""
        
        transaction_costs = {}
        total_cost = 0.0
        
        for symbol, weight_change in weight_changes.items():
            if abs(weight_change) < 1e-6:  # Skip negligible changes
                continue
                
            # Calculate trade value
            trade_value = abs(weight_change * portfolio_value)
            price = current_prices.get(symbol, 100.0)  # Default price if not available
            shares = trade_value / price
            
            # Estimate costs
            commission = trade_value * self.config.commission_rate
            spread_cost = trade_value * self.config.bid_ask_spread / 2
            impact_cost = trade_value * self.config.market_impact
            
            total_symbol_cost = commission + spread_cost + impact_cost
            cost_pct = total_symbol_cost / trade_value if trade_value > 0 else 0
            
            transaction_costs[symbol] = {
                'trade_value': trade_value,
                'shares': shares,
                'commission': commission,
                'spread_cost': spread_cost,
                'impact_cost': impact_cost,
                'total_cost': total_symbol_cost,
                'cost_percentage': cost_pct
            }
            
            total_cost += total_symbol_cost
        
        # Add summary
        transaction_costs['_summary'] = {
            'total_cost': total_cost,
            'cost_as_pct_portfolio': total_cost / portfolio_value if portfolio_value > 0 else 0,
            'symbols_traded': len([s for s in transaction_costs.keys() if s != '_summary'])
        }
        
        return transaction_costs
    
    def optimize_trade_sequence(self, target_weights: Dict[str, float],
                              current_weights: Dict[str, float],
                              current_prices: Dict[str, float],
                              portfolio_value: float) -> List[Dict[str, Any]]:
        """Optimize the sequence of trades for minimal market impact"""
        
        weight_changes = {}
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            change = target_w - current_w
            if abs(change) >= self.config.min_weight_change:
                weight_changes[symbol] = change
        
        if not weight_changes:
            return []
        
        # Estimate transaction costs
        costs = self.estimate_transaction_costs(weight_changes, portfolio_value, current_prices)
        
        # Create trade sequence
        trades = []
        for symbol, change in weight_changes.items():
            if symbol == '_summary':
                continue
                
            cost_info = costs.get(symbol, {})
            
            # Skip if transaction cost is too high
            if cost_info.get('cost_percentage', 0) > self.config.transaction_cost_threshold:
                self.logger.warning(f"Skipping {symbol} trade - cost too high: {cost_info.get('cost_percentage', 0):.2%}")
                continue
                
            trade = {
                'symbol': symbol,
                'weight_change': change,
                'current_weight': current_weights.get(symbol, 0.0),
                'target_weight': target_weights.get(symbol, 0.0),
                'trade_value': cost_info.get('trade_value', 0),
                'estimated_cost': cost_info.get('total_cost', 0),
                'priority': abs(change),  # Larger changes have higher priority
                'action': 'BUY' if change > 0 else 'SELL'
            }
            trades.append(trade)
        
        # Sort by priority (largest changes first) and action (sells before buys to free up cash)
        trades.sort(key=lambda x: (x['action'] == 'BUY', -x['priority']))
        
        return trades


class MarketRegimeDetector:
    """Detect market regimes using volatility and trend analysis"""
    
    def __init__(self):
        self.regimes = ['low_vol_uptrend', 'low_vol_downtrend', 'high_vol_uptrend', 
                       'high_vol_downtrend', 'high_vol_sideways']
        self.current_regime = 'low_vol_uptrend'
        self.regime_history = []
        
    def detect_regime(self, market_data: pd.DataFrame, lookback: int = 60) -> str:
        """Detect current market regime"""
        
        if len(market_data) < lookback:
            return self.current_regime
        
        recent_data = market_data.tail(lookback)
        
        # Calculate volatility (using returns)
        if 'return' in recent_data.columns:
            volatility = recent_data['return'].std()
        else:
            volatility = recent_data['close'].pct_change().std()
        
        # Calculate trend (using moving average slope)
        ma_short = recent_data['close'].rolling(window=10).mean()
        ma_long = recent_data['close'].rolling(window=30).mean()
        trend = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        
        # Classify regime
        vol_threshold = 0.02  # 2% daily volatility threshold
        trend_threshold = 0.05  # 5% trend threshold
        
        if volatility < vol_threshold:
            if trend > trend_threshold:
                regime = 'low_vol_uptrend'
            elif trend < -trend_threshold:
                regime = 'low_vol_downtrend'
            else:
                regime = 'low_vol_sideways'
        else:
            if trend > trend_threshold:
                regime = 'high_vol_uptrend'
            elif trend < -trend_threshold:
                regime = 'high_vol_downtrend'
            else:
                regime = 'high_vol_sideways'
        
        self.current_regime = regime
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'volatility': volatility,
            'trend': trend
        })
        
        return regime

# Portfolio Optimization Module

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple objectives"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.optimization_history = []
        self.logger = logging.getLogger(__name__)
        
    def optimize_portfolio(self, expected_returns: Dict[str, float],
                          risk_data: Dict[str, pd.Series],
                          current_weights: Dict[str, float] = None,
                          method: str = 'max_sharpe') -> Dict[str, float]:
        """Optimize portfolio weights using specified method"""
        
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Prepare expected returns vector
        mu = np.array([expected_returns[s] for s in symbols])
        
        # Prepare covariance matrix
        returns_df = pd.DataFrame({s: risk_data[s] for s in symbols if s in risk_data})
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 20:
            self.logger.warning("Insufficient data for optimization, using equal weights")
            equal_weight = 1.0 / n_assets
            return {s: equal_weight for s in symbols}
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov().values
        
        # Add regularization for numerical stability
        cov_matrix += np.eye(n_assets) * 1e-6
        
        # Current weights (if provided)
        current_w = np.array([current_weights.get(s, 0) for s in symbols]) if current_weights else None
        
        # Optimize based on method
        try:
            if method == 'max_sharpe':
                optimal_weights = self._maximize_sharpe(mu, cov_matrix)
            elif method == 'min_variance':
                optimal_weights = self._minimize_variance(cov_matrix)
            elif method == 'max_diversification':
                optimal_weights = self._maximize_diversification(cov_matrix)
            elif method == 'risk_parity':
                optimal_weights = self._risk_parity(cov_matrix)
            elif method == 'mean_reversion':
                optimal_weights = self._mean_reversion_weights(current_w, mu, cov_matrix)
            else:
                self.logger.warning(f"Unknown optimization method: {method}")
                optimal_weights = self._maximize_sharpe(mu, cov_matrix)
            
            # Convert to dictionary
            result = {symbols[i]: optimal_weights[i] for i in range(n_assets)}
            
            # Store optimization history
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'method': method,
                'weights': result.copy(),
                'expected_return': np.dot(optimal_weights, mu),
                'expected_risk': np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / n_assets
            return {s: equal_weight for s in symbols}
    
    def optimize_with_smooth_rebalancing(self, expected_returns: Dict[str, float],
                                       risk_data: Dict[str, pd.Series],
                                       current_weights: Dict[str, float],
                                       current_prices: Dict[str, float],
                                       portfolio_value: float,
                                       risk_manager: 'RiskManager',
                                       method: str = 'max_sharpe') -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Optimize portfolio with smooth rebalancing considerations"""
        
        # First, get the target optimal weights
        target_weights = self.optimize_portfolio(expected_returns, risk_data, current_weights, method)
        
        if not self.config.smooth_rebalancing:
            # Direct rebalancing without smoothing
            weight_changes = {}
            for symbol in set(current_weights.keys()) | set(target_weights.keys()):
                current_w = current_weights.get(symbol, 0.0)
                target_w = target_weights.get(symbol, 0.0)
                weight_changes[symbol] = target_w - current_w
            
            trades = risk_manager.optimize_trade_sequence(
                target_weights, current_weights, current_prices, portfolio_value
            )
            return target_weights, trades
        
        # Smooth rebalancing
        smooth_weights = risk_manager.calculate_smooth_rebalancing(
            current_weights, target_weights, portfolio_value, current_prices
        )
        
        # Generate optimized trade sequence
        trades = risk_manager.optimize_trade_sequence(
            smooth_weights, current_weights, current_prices, portfolio_value
        )
        
        # Log rebalancing information
        total_target_turnover = sum(abs(target_weights.get(s, 0) - current_weights.get(s, 0)) 
                                  for s in set(current_weights.keys()) | set(target_weights.keys()))
        total_smooth_turnover = sum(abs(smooth_weights.get(s, 0) - current_weights.get(s, 0)) 
                                  for s in set(current_weights.keys()) | set(smooth_weights.keys()))
        
        self.logger.info(f"Rebalancing summary:")
        self.logger.info(f"  Target turnover: {total_target_turnover:.2%}")
        self.logger.info(f"  Smooth turnover: {total_smooth_turnover:.2%}")
        self.logger.info(f"  Number of trades: {len(trades)}")
        
        if trades:
            total_trade_cost = sum(trade.get('estimated_cost', 0) for trade in trades)
            cost_pct = total_trade_cost / portfolio_value if portfolio_value > 0 else 0
            self.logger.info(f"  Estimated transaction costs: ${total_trade_cost:.2f} ({cost_pct:.3%})")
        
        return smooth_weights, trades
    
    def _maximize_sharpe(self, mu: np.ndarray, cov_matrix: np.ndarray, 
                        risk_free_rate: float = 0.02) -> np.ndarray:
        """Maximize Sharpe ratio"""
        n_assets = len(mu)
        
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -(portfolio_return - risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds (no short selling in simulation mode)
        if self.config.simulate_trading:
            bounds = [(0, self.config.position_limit) for _ in range(n_assets)]
        else:
            bounds = [(-self.config.position_limit, self.config.position_limit) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _minimize_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimize portfolio variance"""
        n_assets = len(cov_matrix)
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds
        if self.config.simulate_trading:
            bounds = [(0, self.config.position_limit) for _ in range(n_assets)]
        else:
            bounds = [(-self.config.position_limit, self.config.position_limit) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _maximize_diversification(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Maximize diversification ratio"""
        n_assets = len(cov_matrix)
        
        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(cov_matrix))
        
        def neg_diversification_ratio(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            weighted_avg_vol = np.dot(weights, asset_vols)
            return -weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds
        if self.config.simulate_trading:
            bounds = [(0, self.config.position_limit) for _ in range(n_assets)]
        else:
            bounds = [(-self.config.position_limit, self.config.position_limit) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_diversification_ratio, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity portfolio (equal risk contribution)"""
        n_assets = len(cov_matrix)
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(n_assets)
            risk_contributions = weights * marginal_risk
            target_risk = np.ones(n_assets) / n_assets
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds (risk parity typically long-only)
        bounds = [(0.01, self.config.position_limit) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _mean_reversion_weights(self, current_weights: np.ndarray, 
                               expected_returns: np.ndarray, 
                               cov_matrix: np.ndarray) -> np.ndarray:
        """Mean reversion portfolio (gradually adjust from current weights)"""
        
        if current_weights is None:
            return self._maximize_sharpe(expected_returns, cov_matrix)
        
        # Get optimal weights without transaction costs
        optimal_weights = self._maximize_sharpe(expected_returns, cov_matrix)
        
        # Blend current and optimal weights to reduce turnover
        turnover_penalty = 0.1  # Penalty for deviating from current weights
        blended_weights = (1 - turnover_penalty) * optimal_weights + turnover_penalty * current_weights
        
        # Normalize
        blended_weights = blended_weights / np.sum(blended_weights)
        
        return blended_weights
    
    def calculate_efficient_frontier(self, expected_returns: Dict[str, float],
                                   risk_data: Dict[str, pd.Series],
                                   n_points: int = 20) -> Dict[str, List[float]]:
        """Calculate efficient frontier points"""
        
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {'returns': [], 'risks': [], 'sharpe_ratios': []}
        
        # Prepare data
        mu = np.array([expected_returns[s] for s in symbols])
        returns_df = pd.DataFrame({s: risk_data[s] for s in symbols if s in risk_data})
        returns_df = returns_df.dropna()
        cov_matrix = returns_df.cov().values
        
        # Calculate min and max return portfolios
        min_var_weights = self._minimize_variance(cov_matrix)
        max_ret_weights = np.zeros(n_assets)
        max_ret_weights[np.argmax(mu)] = 1.0  # All in highest return asset
        
        min_ret = np.dot(min_var_weights, mu)
        max_ret = np.dot(max_ret_weights, mu)
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_returns = []
        frontier_risks = []
        frontier_sharpe = []
        
        for target_ret in target_returns:
            try:
                # Minimize variance subject to return constraint
                def portfolio_variance(weights):
                    return np.dot(weights, np.dot(cov_matrix, weights))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                    {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_ret}
                ]
                
                bounds = [(0, self.config.position_limit) for _ in range(n_assets)]
                x0 = np.array([1.0 / n_assets] * n_assets)
                
                result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, mu)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                    
                    frontier_returns.append(portfolio_return)
                    frontier_risks.append(portfolio_risk)
                    frontier_sharpe.append(sharpe_ratio)
                    
            except Exception:
                continue
        
        return {
            'returns': frontier_returns,
            'risks': frontier_risks, 
            'sharpe_ratios': frontier_sharpe
        }

# Execution Handler Module

class ExecutionHandler:
    """Advanced execution engine with IBKR integration and transaction cost analysis"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ib = None
        self.contracts = {}
        self.order_history = []
        self.execution_costs = defaultdict(float)
        self.positions = defaultdict(int)
        self.pending_orders = {}
        self.logger = logging.getLogger(__name__)
        
        # Daily return tracking
        self.last_portfolio_value = None
        self.daily_returns = []
        self.daily_report_file = os.path.join(config.report_dir, 'daily_report.csv')
        
        # Transaction cost parameters (now from config)
        self.commission_rate = config.commission_rate
        self.bid_ask_spread = config.bid_ask_spread
        self.market_impact = config.market_impact
        
        if not self.config.simulate_trading and IBKR_AVAILABLE:
            self._connect_to_ibkr()
    
    def _connect_to_ibkr(self, max_retries: int = 3):
        """Connect to Interactive Brokers API with retry mechanism"""
        for attempt in range(max_retries):
            try:
                self.ib = IB()
                self.logger.info(f"Attempting to connect to IBKR (attempt {attempt + 1}/{max_retries})")
                
                # Test connection
                success = self.ib.connect(self.config.ib_host, self.config.ib_port, clientId=self.config.ib_client_id)
                
                if not success:
                    raise ConnectionError("IBKR connection returned False")
                
                # Configure market data
                self.ib.reqMarketDataType(self.config.market_data_type)
                
                # Create contracts for symbols
                for symbol in self.config.symbols:
                    contract = Stock(symbol, 'SMART', 'USD')
                    self.ib.qualifyContracts(contract)
                    self.contracts[symbol] = contract
                
                self.logger.info(f"  Successfully connected to IBKR (Client ID: {self.config.ib_client_id})")
                self.logger.info(f"   Host: {self.config.ib_host}:{self.config.ib_port}")
                self.logger.info(f"   Paper Trading: {'Yes' if self.config.ib_port == 7497 else 'No'}")
                return True
                
            except Exception as e:
                self.logger.warning(f"IBKR connection attempt {attempt + 1} failed: {e}")
                if self.ib:
                    try:
                        self.ib.disconnect()
                    except:
                        pass
                    self.ib = None
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)  # Wait before retry
                
        # All attempts failed
        self.logger.error(" Failed to connect to IBKR after all attempts")
        self.logger.error("Please ensure:")
        self.logger.error("1. TWS or IB Gateway is running")
        self.logger.error("2. API connections are enabled in TWS settings")
        self.logger.error("3. Correct port number (7497 for Paper, 7496 for Live)")
        self.logger.error("4. Socket port is available")
        return False
    
    def _check_connection(self) -> bool:
        """Check if IBKR connection is still active"""
        if not self.ib:
            return False
        
        try:
            return self.ib.isConnected()
        except:
            return False
    
    def _ensure_connection(self) -> bool:
        """Ensure IBKR connection is active, reconnect if necessary"""
        if self._check_connection():
            return True
        
        self.logger.warning("IBKR connection lost, attempting to reconnect...")
        return self._connect_to_ibkr()
    
    def execute_trades(self, target_positions: Dict[str, int], 
                      current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """Execute trades to reach target positions"""
        
        execution_results = {}
        
        for symbol, target_qty in target_positions.items():
            current_qty = self.positions.get(symbol, 0)
            trade_qty = target_qty - current_qty
            
            if trade_qty == 0:
                continue  # No trade needed
            
            if symbol not in current_prices:
                self.logger.warning(f"No price data for {symbol}, skipping trade")
                continue
            
            price = current_prices[symbol]
            
            # Execute trade
            if self.config.simulate_trading:
                result = self._simulate_trade(symbol, trade_qty, price)
            else:
                result = self._execute_real_trade(symbol, trade_qty, price)
            
            execution_results[symbol] = result
            
            # Update positions if trade was successful
            if result['status'] == 'filled':
                self.positions[symbol] = target_qty
                self._record_execution(symbol, trade_qty, result['executed_price'], result)
        
        return execution_results
    
    def _simulate_trade(self, symbol: str, quantity: int, price: float) -> Dict:
        """Simulate trade execution"""
        
        # Simulate transaction costs
        commission = abs(quantity) * price * self.commission_rate
        spread_cost = abs(quantity) * price * self.bid_ask_spread / 2
        impact_cost = abs(quantity) * price * self.market_impact
        
        total_cost = commission + spread_cost + impact_cost
        executed_price = price * (1 + self.bid_ask_spread/2 if quantity > 0 else 1 - self.bid_ask_spread/2)
        
        result = {
            'status': 'filled',
            'symbol': symbol,
            'quantity': quantity,
            'requested_price': price,
            'executed_price': executed_price,
            'commission': commission,
            'total_cost': total_cost,
            'timestamp': datetime.now(),
            'order_type': 'SIMULATED'
        }
        
        self.execution_costs[symbol] += total_cost
        
        self.logger.info(f"SIMULATED: {symbol} {quantity:+d} shares @ ${executed_price:.2f} (cost: ${total_cost:.2f})")
        
        return result
    
    def _execute_real_trade(self, symbol: str, quantity: int, price: float) -> Dict:
        """Execute real trade through IBKR"""
        
        # Ensure connection is active
        if not self._ensure_connection() or symbol not in self.contracts:
            return {
                'status': 'error',
                'error': 'IBKR not connected or contract not available',
                'symbol': symbol,
                'quantity': quantity
            }
        
        try:
            contract = self.contracts[symbol]
            
            # Create order based on configuration
            if self.config.order_type == 'MARKET':
                order = MarketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
            elif self.config.order_type == 'LIMIT':
                # Use current price as limit price with small buffer
                limit_price = price * 1.001 if quantity > 0 else price * 0.999
                order = LimitOrder('BUY' if quantity > 0 else 'SELL', abs(quantity), limit_price)
            else:
                # Default to market order
                order = MarketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.pending_orders[symbol] = trade
            
            # Wait for execution (with timeout)
            timeout = 30  # 30 seconds
            start_time = time.time()
            
            while not trade.isDone() and (time.time() - start_time) < timeout:
                self.ib.sleep(0.1)
            
            if trade.isDone():
                fill = trade.fills[-1] if trade.fills else None
                
                if fill:
                    executed_price = fill.execution.price
                    commission = fill.commissionReport.commission if fill.commissionReport else 0
                    
                    result = {
                        'status': 'filled',
                        'symbol': symbol,
                        'quantity': quantity,
                        'requested_price': price,
                        'executed_price': executed_price,
                        'commission': commission,
                        'total_cost': commission + abs(quantity) * abs(executed_price - price),
                        'timestamp': datetime.now(),
                        'order_type': self.config.order_type,
                        'trade_id': trade.order.orderId
                    }
                    
                    self.execution_costs[symbol] += result['total_cost']
                    
                    self.logger.info(f"EXECUTED: {symbol} {quantity:+d} shares @ ${executed_price:.2f}")
                    
                else:
                    result = {
                        'status': 'partial',
                        'symbol': symbol,
                        'quantity': quantity,
                        'error': 'Order not completely filled'
                    }
            else:
                # Cancel unfilled order
                self.ib.cancelOrder(order)
                result = {
                    'status': 'timeout',
                    'symbol': symbol,
                    'quantity': quantity,
                    'error': 'Order timed out'
                }
            
            if symbol in self.pending_orders:
                del self.pending_orders[symbol]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return {
                'status': 'error',
                'symbol': symbol,
                'quantity': quantity,
                'error': str(e)
            }
    
    def _record_execution(self, symbol: str, quantity: int, price: float, execution_details: Dict):
        """Record execution details"""
        
        execution_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': execution_details.get('commission', 0),
            'total_cost': execution_details.get('total_cost', 0),
            'status': execution_details.get('status', 'unknown'),
            'order_type': execution_details.get('order_type', 'unknown')
        }
        
        self.order_history.append(execution_record)
    
    def get_execution_summary(self, symbol: str = None) -> Dict:
        """Get execution cost summary"""
        
        if symbol:
            orders = [o for o in self.order_history if o['symbol'] == symbol]
        else:
            orders = self.order_history
        
        if not orders:
            return {}
        
        total_volume = sum(abs(o['quantity']) * o['price'] for o in orders)
        total_commission = sum(o['commission'] for o in orders)
        total_cost = sum(o['total_cost'] for o in orders)
        
        return {
            'total_trades': len(orders),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'total_cost': total_cost,
            'cost_per_trade': total_cost / len(orders) if orders else 0,
            'cost_as_pct_volume': total_cost / total_volume if total_volume > 0 else 0
        }
    
    def get_current_positions(self) -> Dict[str, int]:
        """Get current positions"""
        
        if not self.config.simulate_trading and self.ib:
            try:
                # Fetch real positions from IBKR
                positions = {}
                for position in self.ib.positions():
                    if position.contract.symbol in self.config.symbols:
                        positions[position.contract.symbol] = int(position.position)
                
                # Update internal tracking
                self.positions.update(positions)
                
                return positions
                
            except Exception as e:
                self.logger.error(f"Error fetching positions from IBKR: {e}")
        
        return dict(self.positions)
    
    def update_daily_return(self, current_prices: Dict[str, float]):
        """Update daily return and save to CSV"""
        if not current_prices or not self.positions:
            return
        
        # Calculate current portfolio value
        portfolio_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                portfolio_value += position * current_prices[symbol]
        
        # Calculate daily return if we have previous value
        daily_return = 0.0
        if self.last_portfolio_value is not None and self.last_portfolio_value > 0:
            daily_return = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        
        # Save to CSV file
        try:
            # Create header if file doesn't exist
            if not os.path.exists(self.daily_report_file):
                os.makedirs(os.path.dirname(self.daily_report_file), exist_ok=True)
                with open(self.daily_report_file, 'w') as f:
                    f.write("Date,Portfolio_Value,Daily_Return_Pct,Cash_Value\n")
            
            # Append daily data
            with open(self.daily_report_file, 'a') as f:
                date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{date_str},{portfolio_value:.2f},{daily_return*100:.4f},{portfolio_value:.2f}\n")
            
            self.logger.info(f"Daily return recorded: Portfolio ${portfolio_value:,.2f}, Return {daily_return*100:+.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error saving daily return: {e}")
        
        # Update tracking variables
        self.last_portfolio_value = portfolio_value
        self.daily_returns.append({
            'date': datetime.now(),
            'portfolio_value': portfolio_value,
            'daily_return': daily_return
        })
        
        # Keep only recent history (last 90 days)
        if len(self.daily_returns) > 90:
            self.daily_returns = self.daily_returns[-90:]
    
    def get_daily_return_summary(self) -> Dict[str, float]:
        """Get summary of daily returns"""
        if not self.daily_returns:
            return {'avg_return': 0, 'volatility': 0, 'total_return': 0}
        
        returns = [r['daily_return'] for r in self.daily_returns if r['daily_return'] != 0]
        
        if not returns:
            return {'avg_return': 0, 'volatility': 0, 'total_return': 0}
        
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        total_return = (self.daily_returns[-1]['portfolio_value'] / self.daily_returns[0]['portfolio_value'] - 1) if len(self.daily_returns) > 1 else 0
        
        return {
            'avg_daily_return': avg_return,
            'daily_volatility': volatility,
            'total_return': total_return,
            'annualized_return': avg_return * 252,
            'annualized_volatility': volatility * np.sqrt(252)
        }

    def save_daily_report(self, current_prices: Dict[str, float]):
        """简化的每日报告保存方法 - 包含详细性能指标"""
        try:
            total_value = sum(pos * current_prices.get(sym, 0) for sym, pos in self.positions.items())
            
            if self.last_portfolio_value is not None and self.last_portfolio_value > 0:
                daily_ret = (total_value - self.last_portfolio_value) / self.last_portfolio_value
            else:
                daily_ret = 0.0
                
            # 更新每日回报历史
            self.daily_returns.append({
                'date': datetime.now(),
                'portfolio_value': total_value,
                'daily_return': daily_ret
            })
            
            # 计算性能指标
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            
            if len(self.daily_returns) > 1:
                # 计算夏普比率
                returns = [r['daily_return'] for r in self.daily_returns if r['daily_return'] != 0]
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    volatility = np.std(returns)
                    if volatility > 0:
                        sharpe_ratio = (avg_return * 252) / (volatility * np.sqrt(252))  # 年化夏普比率
                
                # 计算最大回撤
                values = [r['portfolio_value'] for r in self.daily_returns]
                peak = np.maximum.accumulate(values)
                drawdown = (np.array(values) - peak) / peak
                max_drawdown = drawdown.min()
            
            self.last_portfolio_value = total_value

            # 确保报告目录存在
            os.makedirs(os.path.dirname(self.daily_report_file), exist_ok=True)
            
            # 检查文件是否存在，如果不存在则创建表头
            file_exists = os.path.exists(self.daily_report_file)
            
            with open(self.daily_report_file, "a", encoding='utf-8') as f:
                if not file_exists:
                    f.write("Date,Portfolio_Value,Daily_Return_Pct,Sharpe_Ratio,Max_Drawdown_Pct,Cash_Value\n")
                f.write(f"{datetime.now().strftime('%Y-%m-%d')},{total_value:.2f},{daily_ret:.4%},{sharpe_ratio:.2f},{max_drawdown:.2%},{total_value:.2f}\n")

            print(f"今日交易完成，日报已保存到 {self.daily_report_file}")
            print(f"今日收益率: {daily_ret:.2%}, 夏普比率: {sharpe_ratio:.2f}, 最大回撤: {max_drawdown:.2%}")
            
        except Exception as e:
            self.logger.error(f"保存每日报告失败: {e}")

    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib:
            try:
                self.ib.disconnect()
                self.logger.info("Disconnected from IBKR")
            except Exception as e:
                self.logger.error(f"Error disconnecting from IBKR: {e}")

# Reporting and Visualization Module

class ReportGenerator:
    """Comprehensive reporting and visualization system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.daily_reports = []  # Store daily reports
        
    def generate_daily_report(self, trading_engine, date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive daily performance report"""
        
        if date is None:
            date = datetime.now()
            
        date_str = date.strftime('%Y-%m-%d')
        
        daily_report = {
            'date': date_str,
            'timestamp': date.isoformat(),
            'portfolio_summary': {},
            'daily_pnl': {},
            'positions': {},
            'signals': {},
            'trading_activity': {},
            'risk_metrics': {},
            'market_data': {}
        }
        
        try:
            # Portfolio summary
            portfolio_value = trading_engine.risk_manager.portfolio_history[-1] if trading_engine.risk_manager.portfolio_history else trading_engine.config.initial_capital
            
            # Calculate daily return
            daily_return = 0.0
            if len(trading_engine.risk_manager.portfolio_history) > 1:
                prev_value = trading_engine.risk_manager.portfolio_history[-2]
                daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            
            # Calculate total return since inception
            total_return = (portfolio_value - trading_engine.config.initial_capital) / trading_engine.config.initial_capital
            
            daily_report['portfolio_summary'] = {
                'portfolio_value': portfolio_value,
                'initial_capital': trading_engine.config.initial_capital,
                'daily_return': daily_return,
                'daily_return_pct': daily_return * 100,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'cash_position': portfolio_value - sum(
                    pos * trading_engine.data_handler.get_real_time_data().get(symbol, {}).get('price', 0)
                    for symbol, pos in trading_engine.current_positions.items()
                )
            }
            
            # Daily P&L breakdown by position
            daily_report['daily_pnl'] = {}
            for symbol, position in trading_engine.current_positions.items():
                if position != 0:
                    current_price = trading_engine.data_handler.get_real_time_data().get(symbol, {}).get('price', 0)
                    position_value = position * current_price
                    
                    # Calculate daily P&L (simplified - would need previous day's prices for accuracy)
                    daily_pnl = 0.0
                    if len(trading_engine.performance_history) > 1:
                        prev_record = trading_engine.performance_history[-2]
                        prev_position = prev_record.get('positions', {}).get(symbol, 0)
                        if prev_position != 0:
                            # Estimate daily P&L based on position changes
                            daily_pnl = (position - prev_position) * current_price
                    
                    daily_report['daily_pnl'][symbol] = {
                        'position': position,
                        'current_price': current_price,
                        'position_value': position_value,
                        'daily_pnl': daily_pnl,
                        'weight': position_value / portfolio_value if portfolio_value > 0 else 0
                    }
            
            # Current positions
            daily_report['positions'] = trading_engine.current_positions.copy()
            
            # Latest signals
            daily_report['signals'] = trading_engine.current_signals.copy()
            
            # Trading activity for the day
            today_trades = [
                trade for trade in trading_engine.execution_handler.order_history
                if trade['timestamp'].date() == date.date()
            ]
            
            daily_report['trading_activity'] = {
                'trades_count': len(today_trades),
                'total_volume': sum(abs(trade['quantity']) * trade['price'] for trade in today_trades),
                'total_commission': sum(trade.get('commission', 0) for trade in today_trades),
                'trades_detail': today_trades
            }
            
            # Risk metrics
            if trading_engine.risk_manager.risk_metrics_history:
                latest_risk = trading_engine.risk_manager.risk_metrics_history[-1]
                daily_report['risk_metrics'] = {
                    'var_95': latest_risk.get('var_95', 0),
                    'expected_shortfall': latest_risk.get('expected_shortfall', 0),
                    'max_position_weight': latest_risk.get('max_position_weight', 0),
                    'total_exposure': latest_risk.get('total_exposure', 0),
                    'diversification_ratio': latest_risk.get('diversification_ratio', 1.0)
                }
            
            # Market data summary
            market_prices = trading_engine.data_handler.get_real_time_data()
            daily_report['market_data'] = {
                symbol: {
                    'price': data.get('price', 0),
                    'signal': trading_engine.current_signals.get(symbol, 0.5)
                }
                for symbol, data in market_prices.items()
            }
            
            # Performance metrics
            if len(trading_engine.performance_history) > 1:
                portfolio_values = [record['portfolio_value'] for record in trading_engine.performance_history]
                returns = pd.Series(portfolio_values).pct_change().dropna()
                
                if len(returns) > 0:
                    daily_report['performance_metrics'] = {
                        'volatility': returns.std() * np.sqrt(252),  # Annualized
                        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                        'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                        'win_rate': (returns > 0).mean(),
                        'avg_return': returns.mean(),
                        'best_day': returns.max(),
                        'worst_day': returns.min()
                    }
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            daily_report['error'] = str(e)
        
        # Store the daily report
        self.daily_reports.append(daily_report)
        
        return daily_report
    
    def save_daily_report(self, daily_report: Dict[str, Any], filename: str = None):
        """Save daily report to file"""
        
        if filename is None:
            date_str = daily_report.get('date', datetime.now().strftime('%Y-%m-%d'))
            filename = f'daily_report_{date_str}.json'
        
        filepath = os.path.join(self.config.report_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(daily_report, f, indent=2, default=str)
            self.logger.info(f"Daily report saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save daily report: {e}")
            return None
    
    def create_daily_report_html(self, daily_report: Dict[str, Any]) -> str:
        """Create HTML version of daily report"""
        
        date_str = daily_report.get('date', 'Unknown')
        portfolio_summary = daily_report.get('portfolio_summary', {})
        daily_pnl = daily_report.get('daily_pnl', {})
        trading_activity = daily_report.get('trading_activity', {})
        risk_metrics = daily_report.get('risk_metrics', {})
        performance_metrics = daily_report.get('performance_metrics', {})
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Daily Trading Report - {date_str}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #f39c12; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .status-good {{ background-color: #d5edda; color: #155724; }}
        .status-warning {{ background-color: #fff3cd; color: #856404; }}
        .status-danger {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Daily Trading Report</h1>
            <h2>{date_str}</h2>
            <p>Generated at: {daily_report.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h3>Portfolio Summary</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">${portfolio_summary.get('portfolio_value', 0):,.2f}</div>
                    <div class="metric-label">Portfolio Value</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if portfolio_summary.get('daily_return_pct', 0) > 0 else 'negative' if portfolio_summary.get('daily_return_pct', 0) < 0 else 'neutral'}">{portfolio_summary.get('daily_return_pct', 0):+.2f}%</div>
                    <div class="metric-label">Daily Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if portfolio_summary.get('total_return_pct', 0) > 0 else 'negative' if portfolio_summary.get('total_return_pct', 0) < 0 else 'neutral'}">{portfolio_summary.get('total_return_pct', 0):+.2f}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${portfolio_summary.get('cash_position', 0):,.2f}</div>
                    <div class="metric-label">Cash Position</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Daily P&L Breakdown</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Position</th>
                        <th>Current Price</th>
                        <th>Position Value</th>
                        <th>Daily P&L</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for symbol, pnl_data in daily_pnl.items():
            daily_pnl_value = pnl_data.get('daily_pnl', 0)
            pnl_class = 'positive' if daily_pnl_value > 0 else 'negative' if daily_pnl_value < 0 else 'neutral'
            
            html_content += f"""
                    <tr>
                        <td><strong>{symbol}</strong></td>
                        <td>{pnl_data.get('position', 0):,}</td>
                        <td>${pnl_data.get('current_price', 0):.2f}</td>
                        <td>${pnl_data.get('position_value', 0):,.2f}</td>
                        <td class="{pnl_class}">${daily_pnl_value:+,.2f}</td>
                        <td>{pnl_data.get('weight', 0):.1%}</td>
                    </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h3>Trading Activity</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{trading_activity.get('trades_count', 0)}</div>
                    <div class="metric-label">Trades Executed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${trading_activity.get('total_volume', 0):,.2f}</div>
                    <div class="metric-label">Total Volume</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${trading_activity.get('total_commission', 0):,.2f}</div>
                    <div class="metric-label">Total Commission</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>Risk Metrics</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('var_95', 0):.3f}</div>
                    <div class="metric-label">VaR (95%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('max_position_weight', 0):.1%}</div>
                    <div class="metric-label">Max Position Weight</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('total_exposure', 0):.1%}</div>
                    <div class="metric-label">Total Exposure</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('diversification_ratio', 1):.2f}</div>
                    <div class="metric-label">Diversification Ratio</div>
                </div>
            </div>
        </div>
        """
        
        if performance_metrics:
            html_content += f"""
        <div class="section">
            <h3>Performance Metrics</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics.get('volatility', 0):.1%}</div>
                    <div class="metric-label">Volatility (Ann.)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics.get('max_drawdown', 0):.1%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics.get('win_rate', 0):.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
        </div>
            """
        
        html_content += """
        <div class="section">
            <h3>Current Signals</h3>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Signal</th>
                        <th>Action</th>
                        <th>Current Price</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        signals = daily_report.get('signals', {})
        market_data = daily_report.get('market_data', {})
        
        for symbol in signals.keys():
            signal_value = signals.get(symbol, 0.5)
            price = market_data.get(symbol, {}).get('price', 0)
            
            if signal_value >= 0.65:
                action = "BUY"
                action_class = "status-good"
            elif signal_value <= 0.35:
                action = "SELL"
                action_class = "status-danger"
            else:
                action = "HOLD"
                action_class = "status-warning"
            
            html_content += f"""
                    <tr>
                        <td><strong>{symbol}</strong></td>
                        <td>{signal_value:.3f}</td>
                        <td class="{action_class}">{action}</td>
                        <td>${price:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <p style="text-align: center; color: #7f8c8d; font-size: 12px;">
                Generated by Quant Trading Engine Plus | """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def save_daily_report_html(self, daily_report: Dict[str, Any], filename: str = None) -> str:
        """Save daily report as HTML file"""
        
        if filename is None:
            date_str = daily_report.get('date', datetime.now().strftime('%Y-%m-%d'))
            filename = f'daily_report_{date_str}.html'
        
        filepath = os.path.join(self.config.report_dir, filename)
        
        try:
            html_content = self.create_daily_report_html(daily_report)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            self.logger.info(f"Daily HTML report saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save daily HTML report: {e}")
            return None
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return drawdown.min()
        
    def generate_performance_report(self, trading_engine) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'summary': {},
            'positions': {},
            'signals': {},
            'risk_metrics': {},
            'execution_summary': {}
        }
        
        try:
            # Current positions and portfolio value
            positions = trading_engine.execution_handler.get_current_positions()
            prices = trading_engine.data_handler.get_real_time_data()
            
            portfolio_value = sum(pos * prices.get(symbol, {}).get('price', 0) 
                                for symbol, pos in positions.items())
            
            report['summary'] = {
                'portfolio_value': portfolio_value,
                'total_positions': len([p for p in positions.values() if p != 0]),
                'cash_equivalent': self.config.initial_capital - portfolio_value
            }
            
            report['positions'] = positions
            
            # Signal analysis
            if hasattr(trading_engine, 'signal_generator') and trading_engine.signal_generator:
                for symbol in self.config.symbols:
                    signal_analysis = trading_engine.signal_generator.get_signal_analysis(symbol)
                    if signal_analysis:
                        report['signals'][symbol] = signal_analysis
            
            # Risk metrics
            if hasattr(trading_engine, 'risk_manager') and trading_engine.risk_manager:
                if trading_engine.risk_manager.risk_metrics_history:
                    latest_risk = trading_engine.risk_manager.risk_metrics_history[-1]
                    report['risk_metrics'] = latest_risk
            
            # Execution summary
            report['execution_summary'] = trading_engine.execution_handler.get_execution_summary()
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            report['error'] = str(e)
        
        return report
    
    def save_report_to_file(self, report: Dict[str, Any], filename: str = None):
        """Save report to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_report_{timestamp}.json"
        
        filepath = os.path.join(self.config.report_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def create_performance_plots(self, trading_engine) -> List[str]:
        """Create performance visualization plots"""
        
        if not PLOTTING_AVAILABLE and not PLOTLY_AVAILABLE:
            self.logger.warning("Plotting libraries not available")
            return []
        
        plot_files = []
        
        try:
            # Portfolio value over time
            if trading_engine.risk_manager.portfolio_history:
                plot_file = self._plot_portfolio_evolution(trading_engine.risk_manager.portfolio_history)
                if plot_file:
                    plot_files.append(plot_file)
            
            # Signal analysis plots
            for symbol in self.config.symbols:
                if hasattr(trading_engine.signal_generator, 'signal_history'):
                    signal_history = trading_engine.signal_generator.signal_history.get(symbol, [])
                    if signal_history:
                        plot_file = self._plot_signal_evolution(symbol, signal_history)
                        if plot_file:
                            plot_files.append(plot_file)
            
            # Risk metrics over time
            if trading_engine.risk_manager.risk_metrics_history:
                plot_file = self._plot_risk_metrics(trading_engine.risk_manager.risk_metrics_history)
                if plot_file:
                    plot_files.append(plot_file)
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
        
        return plot_files
    
    def _plot_portfolio_evolution(self, portfolio_history: List[float]) -> Optional[str]:
        """Plot portfolio value evolution"""
        
        if not portfolio_history:
            return None
        
        filename = os.path.join(self.config.report_dir, 'portfolio_evolution.png')
        
        try:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=portfolio_history,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title='Portfolio Value Evolution',
                    xaxis_title='Time Period',
                    yaxis_title='Portfolio Value ($)',
                    template='plotly_white'
                )
                
                fig.write_image(filename)
                
            elif PLOTTING_AVAILABLE:
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_history, linewidth=2, color='blue')
                plt.title('Portfolio Value Evolution')
                plt.xlabel('Time Period')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error plotting portfolio evolution: {e}")
            return None
    
    def _plot_signal_evolution(self, symbol: str, signal_history: List[Dict]) -> Optional[str]:
        """Plot signal evolution for a symbol"""
        
        if not signal_history or len(signal_history) < 2:
            return None
        
        filename = os.path.join(self.config.report_dir, f'{symbol}_signals.png')
        
        try:
            timestamps = [h['timestamp'] for h in signal_history]
            fused_signals = [h['fused_signal'] for h in signal_history]
            ml_signals = [h['ml_signal'] for h in signal_history if h['ml_signal'] is not None]
            technical_signals = [h['technical_signal'] for h in signal_history if h['technical_signal'] is not None]
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(rows=1, cols=1)
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=fused_signals,
                    mode='lines',
                    name='Fused Signal',
                    line=dict(color='black', width=2)
                ))
                
                if len(ml_signals) == len(timestamps):
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=ml_signals,
                        mode='lines',
                        name='ML Signal',
                        line=dict(color='blue', width=1)
                    ))
                
                if len(technical_signals) == len(timestamps):
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=technical_signals,
                        mode='lines',
                        name='Technical Signal',
                        line=dict(color='red', width=1)
                    ))
                
                # Add threshold lines
                fig.add_hline(y=self.config.signal_threshold_buy, line_dash="dash", 
                             line_color="green", annotation_text="Buy Threshold")
                fig.add_hline(y=self.config.signal_threshold_sell, line_dash="dash", 
                             line_color="red", annotation_text="Sell Threshold")
                
                fig.update_layout(
                    title=f'{symbol} Signal Evolution',
                    xaxis_title='Time',
                    yaxis_title='Signal Strength',
                    template='plotly_white'
                )
                
                fig.write_image(filename)
                
            elif PLOTTING_AVAILABLE:
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps, fused_signals, linewidth=2, color='black', label='Fused Signal')
                
                if len(ml_signals) == len(timestamps):
                    plt.plot(timestamps, ml_signals, linewidth=1, color='blue', alpha=0.7, label='ML Signal')
                
                if len(technical_signals) == len(timestamps):
                    plt.plot(timestamps, technical_signals, linewidth=1, color='red', alpha=0.7, label='Technical Signal')
                
                plt.axhline(y=self.config.signal_threshold_buy, color='green', linestyle='--', alpha=0.7, label='Buy Threshold')
                plt.axhline(y=self.config.signal_threshold_sell, color='red', linestyle='--', alpha=0.7, label='Sell Threshold')
                
                plt.title(f'{symbol} Signal Evolution')
                plt.xlabel('Time')
                plt.ylabel('Signal Strength')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error plotting signal evolution for {symbol}: {e}")
            return None
    
    def _plot_risk_metrics(self, risk_history: List[Dict]) -> Optional[str]:
        """Plot risk metrics over time"""
        
        if not risk_history:
            return None
        
        filename = os.path.join(self.config.report_dir, 'risk_metrics.png')
        
        try:
            timestamps = [r['timestamp'] for r in risk_history]
            var_values = [r.get('var_95', np.nan) for r in risk_history]
            diversification_ratios = [r.get('diversification_ratio', np.nan) for r in risk_history]
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Value at Risk (95%)', 'Diversification Ratio'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=var_values,
                    mode='lines',
                    name='VaR 95%',
                    line=dict(color='red', width=2)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=diversification_ratios,
                    mode='lines',
                    name='Diversification Ratio',
                    line=dict(color='blue', width=2)
                ), row=2, col=1)
                
                fig.update_layout(
                    title='Risk Metrics Evolution',
                    template='plotly_white',
                    height=600
                )
                
                fig.write_image(filename)
                
            elif PLOTTING_AVAILABLE:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                ax1.plot(timestamps, var_values, linewidth=2, color='red')
                ax1.set_title('Value at Risk (95%)')
                ax1.set_ylabel('VaR')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(timestamps, diversification_ratios, linewidth=2, color='blue')
                ax2.set_title('Diversification Ratio')
                ax2.set_ylabel('Ratio')
                ax2.set_xlabel('Time')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error plotting risk metrics: {e}")
            return None

# ============================================================================
# Main Trading Engine Class
# ============================================================================

class QuantTradingEnginePlus:
    """Enhanced Quantitative Trading Engine with Modular Architecture"""
    
    def __init__(self, config: TradingConfig = None):
        """Initialize the trading engine"""
        
        self.config = config or TradingConfig()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.report_dir, 'trading_engine.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.signal_generator = SignalGenerator(self.config, self.data_handler)
        self.risk_manager = RiskManager(self.config)
        self.portfolio_optimizer = PortfolioOptimizer(self.config)
        self.execution_handler = ExecutionHandler(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # State variables
        self.is_running = False
        self.historical_data = {}
        self.current_signals = {}
        self.current_positions = {}
        self.last_trade_time = {}
        self.performance_history = []
        
        # Trading schedule control
        self.last_rebalance_date = None
        self.daily_rebalance_hour = 15  # 3 PM market close time
        self.trading_days_only = True  # Only trade on weekdays
        self.last_daily_report_date = None  # Track when last daily report was generated
        
        self.logger.info("QuantTradingEnginePlus initialized successfully")
    
    def initialize(self):
        """Initialize the trading engine with data and models"""
        
        self.logger.info("Initializing trading engine...")
        
        try:
            # Load historical data
            self.logger.info("Loading historical data...")
            self.historical_data = self.data_handler.load_historical_data(
                symbols=self.config.symbols
            )
            
            if not self.historical_data:
                raise ValueError("No historical data loaded")
            
            self.logger.info(f"Loaded data for {len(self.historical_data)} symbols")
            
            # Train ML models
            self.logger.info("Training ML models...")
            model_results = self.signal_generator.train_models(self.historical_data)
            
            self.logger.info(f"Trained models for {len(model_results)} symbols")
            for symbol, result in model_results.items():
                self.logger.info(f"{symbol}: {result['model_type']} (CV Score: {result['cv_score']:.4f})")
            
            # Initialize positions
            self.current_positions = self.execution_handler.get_current_positions()
            self.logger.info(f"Current positions: {self.current_positions}")
            
            # Initialize daily return tracking with current portfolio value
            current_prices = self.data_handler.get_real_time_data()
            if current_prices and self.current_positions:
                prices_dict = {symbol: data['price'] for symbol, data in current_prices.items()}
                initial_portfolio_value = sum(
                    pos * prices_dict.get(symbol, 0) 
                    for symbol, pos in self.current_positions.items()
                )
                self.execution_handler.last_portfolio_value = initial_portfolio_value
                self.logger.info(f"Initial portfolio value: ${initial_portfolio_value:,.2f}")
            
            # Initialize last trade times
            self.last_trade_time = {symbol: 0 for symbol in self.config.symbols}
            
            self.logger.info("Trading engine initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {e}")
            raise
    
    def run_strategy(self, max_iterations: int = None, run_time_seconds: int = None):
        """Run the trading strategy"""
        
        if not self.historical_data:
            raise RuntimeError("Trading engine not initialized. Call initialize() first.")
        
        self.logger.info("Starting trading strategy execution")
        self.is_running = True
        
        iteration = 0
        start_time = time.time()
        
        try:
            while self.is_running:
                iteration += 1
                
                # Check termination conditions
                if max_iterations and iteration > max_iterations:
                    self.logger.info(f"Reached maximum iterations ({max_iterations})")
                    break
                
                if run_time_seconds and (time.time() - start_time) > run_time_seconds:
                    self.logger.info(f"Reached maximum runtime ({run_time_seconds} seconds)")
                    break
                
                # Execute one trading cycle
                self._execute_trading_cycle(iteration)
                
                # Generate daily report if needed
                self._check_and_generate_daily_report()
                
                # Wait for next cycle
                time.sleep(self.config.refresh_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            raise
        finally:
            self.stop()
    
    def _should_rebalance(self) -> bool:
        """Check if it's time for daily rebalancing"""
        now = datetime.now()
        
        # Check if it's a trading day (weekday)
        if self.trading_days_only and now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if it's after rebalancing hour
        if now.hour < self.daily_rebalance_hour:
            return False
        
        # Check if we already rebalanced today
        current_date = now.date()
        if self.last_rebalance_date == current_date:
            return False
        
        return True
    
    def _check_and_generate_daily_report(self):
        """Check if we need to generate a daily report and do so if needed"""
        now = datetime.now()
        current_date = now.date()
        
        # Generate daily report at market close (4:30 PM) or if it's a new day
        should_generate_report = False
        
        # Check if it's a new trading day and we haven't generated a report yet
        if self.last_daily_report_date != current_date:
            # Only generate for weekdays (trading days)
            if current_date.weekday() < 5:
                # Generate report after market close (4:30 PM) or if it's already past that time
                if now.hour >= 16 and now.minute >= 30:
                    should_generate_report = True
                # Or if it's the next day and we missed the close
                elif self.last_daily_report_date is not None and self.last_daily_report_date < current_date:
                    should_generate_report = True
        
        if should_generate_report:
            try:
                self.logger.info(f"Generating daily report for {current_date}")
                daily_report = self.report_generator.generate_daily_report(self, now)
                
                # Save both JSON and HTML versions
                json_file = self.report_generator.save_daily_report(daily_report)
                html_file = self.report_generator.save_daily_report_html(daily_report)
                
                self.last_daily_report_date = current_date
                
                # Log summary
                portfolio_summary = daily_report.get('portfolio_summary', {})
                daily_return_pct = portfolio_summary.get('daily_return_pct', 0)
                portfolio_value = portfolio_summary.get('portfolio_value', 0)
                
                self.logger.info(f"Daily Report Summary:")
                self.logger.info(f"   Portfolio Value: ${portfolio_value:,.2f}")
                self.logger.info(f"   Daily Return: {daily_return_pct:+.2f}%")
                self.logger.info(f"   JSON Report: {json_file}")
                self.logger.info(f"   HTML Report: {html_file}")
                
                # Print a nice summary to console
                self._print_daily_report_summary(daily_report)
                
            except Exception as e:
                self.logger.error(f"Failed to generate daily report: {e}")
    
    def _print_daily_report_summary(self, daily_report: Dict[str, Any]):
        """Print a formatted daily report summary to console"""
        
        print("\n" + "="*80)
        print("DAILY TRADING REPORT SUMMARY")
        print("="*80)
        
        # Portfolio summary
        portfolio_summary = daily_report.get('portfolio_summary', {})
        print(f"Date: {daily_report.get('date', 'Unknown')}")
        print(f"Portfolio Value: ${portfolio_summary.get('portfolio_value', 0):,.2f}")
        
        daily_return_pct = portfolio_summary.get('daily_return_pct', 0)
        return_indicator = "UP" if daily_return_pct > 0 else "DOWN" if daily_return_pct < 0 else "FLAT"
        print(f"{return_indicator} Daily Return: {daily_return_pct:+.2f}%")
        
        total_return_pct = portfolio_summary.get('total_return_pct', 0)
        total_indicator = "PROFIT" if total_return_pct > 0 else "LOSS" if total_return_pct < 0 else "NEUTRAL"
        print(f"{total_indicator} Total Return: {total_return_pct:+.2f}%")
        
        # Top positions
        daily_pnl = daily_report.get('daily_pnl', {})
        if daily_pnl:
            print(f"\nTop Positions:")
            sorted_positions = sorted(daily_pnl.items(), 
                                    key=lambda x: abs(x[1].get('position_value', 0)), 
                                    reverse=True)
            
            for symbol, pnl_data in sorted_positions[:5]:  # Top 5 positions
                position_value = pnl_data.get('position_value', 0)
                weight = pnl_data.get('weight', 0)
                daily_pnl_value = pnl_data.get('daily_pnl', 0)
                pnl_indicator = "UP" if daily_pnl_value > 0 else "DOWN" if daily_pnl_value < 0 else "FLAT"
                
                print(f"   {symbol}: ${position_value:,.2f} ({weight:.1%}) {pnl_indicator} ${daily_pnl_value:+,.2f}")
        
        # Trading activity
        trading_activity = daily_report.get('trading_activity', {})
        trades_count = trading_activity.get('trades_count', 0)
        if trades_count > 0:
            total_volume = trading_activity.get('total_volume', 0)
            total_commission = trading_activity.get('total_commission', 0)
            print(f"\nTrading Activity:")
            print(f"   Trades: {trades_count}")
            print(f"   Volume: ${total_volume:,.2f}")
            print(f"   Commission: ${total_commission:.2f}")
        
        # Current signals
        signals = daily_report.get('signals', {})
        if signals:
            print(f"\nCurrent Signals:")
            for symbol, signal in signals.items():
                if signal >= 0.65:
                    action = "BUY"
                elif signal <= 0.35:
                    action = "SELL"
                else:
                    action = "HOLD"
                print(f"   {symbol}: {signal:.3f} ({action})")
        
        # Risk metrics
        risk_metrics = daily_report.get('risk_metrics', {})
        if risk_metrics:
            print(f"\nRisk Metrics:")
            print(f"   VaR (95%): {risk_metrics.get('var_95', 0):.3f}")
            print(f"   Max Position: {risk_metrics.get('max_position_weight', 0):.1%}")
            print(f"   Total Exposure: {risk_metrics.get('total_exposure', 0):.1%}")
        
        # Daily return summary
        try:
            return_summary = self.execution_handler.get_daily_return_summary()
            if return_summary and return_summary.get('avg_daily_return', 0) != 0:
                print(f"\nDaily Return Summary:")
                print(f"   Avg Daily Return: {return_summary.get('avg_daily_return', 0)*100:+.2f}%")
                print(f"   Total Return: {return_summary.get('total_return', 0)*100:+.2f}%")
                print(f"   Daily Volatility: {return_summary.get('daily_volatility', 0)*100:.2f}%")
        except Exception as e:
            print(f"   Error getting return summary: {e}")
        
        print("="*80)
        print(f"Full reports saved to: {self.config.report_dir}/daily_report_{daily_report.get('date', 'unknown')}.*")
        print(f"Daily returns CSV: {self.execution_handler.daily_report_file}")
        print("="*80 + "\n")

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:30 AM - 4:00 PM ET)"""
        now = datetime.now()
        
        # Simple check for weekdays
        if now.weekday() >= 5:  # Weekend
            return False
        
        hour = now.hour
        minute = now.minute
        
        # Market opens at 9:30 AM, closes at 4:00 PM
        market_open = 9.5  # 9:30 AM
        market_close = 16.0  # 4:00 PM
        current_time = hour + minute / 60.0
        
        return market_open <= current_time <= market_close
        
    def _execute_trading_cycle(self, iteration: int):
        """Execute one complete trading cycle"""
        
        cycle_start = time.time()
        self.logger.info(f"=== Trading Cycle {iteration} ===")
        
        # Check if we should skip this cycle due to timing
        if not self._is_market_hours() and not self.config.simulate_trading:
            self.logger.info("Outside market hours, skipping cycle")
            return
        
        # Check if it's time for daily rebalancing
        should_rebalance = self._should_rebalance()
        if not should_rebalance and not self.config.simulate_trading:
            self.logger.info("Not time for daily rebalancing, monitoring only")
            # Still update metrics but don't trade
            self._update_portfolio_metrics_only()
            return
        
        try:
            # 1. Get current market data
            current_prices = self.data_handler.get_real_time_data()
            if not current_prices:
                self.logger.warning("No current price data available")
                return
            
            # 2. Update historical data with latest prices (simulate real-time updates)
            self._update_historical_data_with_current_prices(current_prices)
            
            # 3. Generate signals
            self.logger.debug("Generating signals...")
            
            ml_signals = self.signal_generator.generate_ml_signals(self.historical_data)
            technical_signals = self.signal_generator.generate_technical_signals(self.historical_data)
            volatility_signals = self.signal_generator.generate_volatility_signals(self.historical_data)
            
            # Fuse signals
            self.current_signals = self.signal_generator.fuse_signals(
                ml_signals, technical_signals, volatility_signals
            )
            
            self.logger.info(f"Signals: {self.current_signals}")
            
            # 4. Risk adjustment
            for symbol in self.current_signals:
                adjusted_signal = self.risk_manager.get_risk_adjusted_signal(
                    self.current_signals[symbol], 
                    symbol, 
                    self.current_positions, 
                    {s: data['price'] for s, data in current_prices.items()}
                )
                self.current_signals[symbol] = adjusted_signal
            
            # 5. Portfolio optimization
            self.logger.debug("Optimizing portfolio...")
            
            # Convert signals to expected returns (simple mapping for demonstration)
            expected_returns = {symbol: (signal - 0.5) * 0.1 for symbol, signal in self.current_signals.items()}
            
            # Extract return series for optimization
            returns_data = {symbol: df['return'] for symbol, df in self.historical_data.items()}
            
            # Get current weights
            portfolio_value = sum(pos * current_prices.get(symbol, {}).get('price', 0) 
                                for symbol, pos in self.current_positions.items())
            
            if portfolio_value <= 0:
                portfolio_value = self.config.initial_capital
            
            current_weights = {}
            if portfolio_value > 0:
                current_weights = {
                    symbol: (pos * current_prices.get(symbol, {}).get('price', 0)) / portfolio_value
                    for symbol, pos in self.current_positions.items()
                }
            
            # Optimize portfolio
            optimal_weights = self.portfolio_optimizer.optimize_portfolio(
                expected_returns, returns_data, current_weights, method='max_sharpe'
            )
            
            self.logger.debug(f"Optimal weights: {optimal_weights}")
            
            # 6. Calculate target positions
            prices_dict = {symbol: data['price'] for symbol, data in current_prices.items()}
            
            target_positions = self.risk_manager.calculate_position_sizes(
                self.current_signals,
                prices_dict,
                portfolio_value,
                self.current_positions
            )
            
            self.logger.info(f"Target positions: {target_positions}")
            
            # 7. Risk checks
            risk_checks = self.risk_manager.check_risk_limits(
                target_positions, prices_dict, returns_data
            )
            
            risk_violations = [check for check, passed in risk_checks.items() if not passed]
            if risk_violations:
                self.logger.warning(f"Risk limit violations: {risk_violations}")
                # Could implement risk override logic here
            
            # 8. Execute trades (only if rebalancing is needed)
            if should_rebalance:
                trades_to_execute = {}
                
                for symbol, target_qty in target_positions.items():
                    current_qty = self.current_positions.get(symbol, 0)
                    
                    if target_qty != current_qty:
                        trades_to_execute[symbol] = target_qty
                
                if trades_to_execute:
                    self.logger.info(f"Executing daily rebalance: {trades_to_execute}")
                    
                    execution_results = self.execution_handler.execute_trades(
                        trades_to_execute, prices_dict
                    )
                    
                    # Update positions and mark rebalance as done
                    for symbol, result in execution_results.items():
                        if result['status'] == 'filled':
                            self.current_positions[symbol] = trades_to_execute[symbol]
                            self.logger.info(f"Rebalanced: {symbol} -> {trades_to_execute[symbol]} shares")
                    
                    # Mark today as rebalanced
                    self.last_rebalance_date = datetime.now().date()
                    
                else:
                    self.logger.info("No rebalancing needed - portfolio already optimal")
            
            # 9. Update risk metrics
            self.risk_manager.update_risk_metrics(
                portfolio_value, self.current_positions, prices_dict, returns_data
            )
            
            # 10. Record performance
            self._record_performance(iteration, cycle_start, portfolio_value)
            
            # 11. Update daily return tracking
            if prices_dict:
                self.execution_handler.update_daily_return(prices_dict)
            
            # 12. Print daily performance summary
            self._print_performance_summary(portfolio_value)
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle {iteration}: {e}")
    
    def _update_portfolio_metrics_only(self):
        """Update portfolio metrics without trading"""
        try:
            current_prices = self.data_handler.get_real_time_data()
            if current_prices:
                prices_dict = {symbol: data['price'] for symbol, data in current_prices.items()}
                portfolio_value = sum(pos * prices_dict.get(symbol, 0) 
                                    for symbol, pos in self.current_positions.items())
                
                if portfolio_value > 0:
                    returns_data = {symbol: df['return'] for symbol, df in self.historical_data.items()}
                    self.risk_manager.update_risk_metrics(
                        portfolio_value, self.current_positions, prices_dict, returns_data
                    )
                    
                    self.logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
                    
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _update_historical_data_with_current_prices(self, current_prices: Dict[str, Dict]):
        """Simulate updating historical data with current prices"""
        
        for symbol, price_data in current_prices.items():
            if symbol in self.historical_data:
                # Create a new row with current price
                df = self.historical_data[symbol]
                if not df.empty:
                    last_date = pd.to_datetime(df['date'].iloc[-1])
                    new_date = last_date + pd.Timedelta(minutes=self.config.refresh_interval // 60)
                    
                    current_price = price_data['price']
                    last_price = df['close'].iloc[-1]
                    
                    new_row = {
                        'date': new_date,
                        'close': current_price,
                        'return': (current_price / last_price - 1) if last_price > 0 else 0
                    }
                    
                    # Add the new row (in practice, this would come from real data feed)
                    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Keep only recent data to avoid memory issues
                    if len(new_df) > self.config.lookback_days * 2:
                        new_df = new_df.tail(self.config.lookback_days * 2).reset_index(drop=True)
                    
                    # Recalculate technical indicators
                    self.historical_data[symbol] = self.data_handler._add_technical_indicators(new_df)
    
    def _print_performance_summary(self, portfolio_value: float):
        """Print daily performance summary"""
        
        if not self.performance_history:
            return
        
        # Calculate daily performance metrics
        try:
            initial_value = self.config.initial_capital
            total_return = (portfolio_value / initial_value - 1) * 100
            
            # Get portfolio returns for metrics calculation
            portfolio_values = [record['portfolio_value'] for record in self.performance_history]
            
            if len(portfolio_values) > 1:
                portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
                
                if len(portfolio_returns) > 0:
                    portfolio_metrics = PerformanceMetrics.calculate_returns_metrics(portfolio_returns)
                    
                    # Print summary
                    print("\n" + "="*60)
                    print("DAILY PERFORMANCE SUMMARY")
                    print("="*60)
                    print(f"Portfolio Value: ${portfolio_value:,.2f}")
                    print(f"Total Return: {total_return:+.2f}%")
                    print(f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}")
                    print(f"Win Rate: {portfolio_metrics.get('win_rate', 0):.1%}")
                    print(f"Volatility: {portfolio_metrics.get('annualized_volatility', 0):.1%}")
                    
                    # Current positions
                    if self.current_positions:
                        print(f"\nCurrent Positions:")
                        for symbol, qty in self.current_positions.items():
                            if qty != 0:
                                print(f"   {symbol}: {qty:,} shares")
                    
                    # Latest signals
                    if self.current_signals:
                        print(f"\nLatest Signals:")
                        for symbol, signal in self.current_signals.items():
                            action = "BUY" if signal >= self.config.signal_threshold_buy else \
                                    "SELL" if signal <= self.config.signal_threshold_sell else \
                                    "HOLD"
                            print(f"   {symbol}: {signal:.3f} ({action})")
                    
                    print("="*60)
                    
                    # Save metrics to JSON for easy viewing
                    metrics_summary = {
                        'timestamp': datetime.now().isoformat(),
                        'portfolio_value': portfolio_value,
                        'total_return_pct': total_return,
                        'performance_metrics': portfolio_metrics,
                        'positions': self.current_positions,
                        'signals': self.current_signals
                    }
                    
                    import json
                    metrics_file = os.path.join(self.config.report_dir, 'daily_performance.json')
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics_summary, f, indent=2, default=str)
                        
        except Exception as e:
            self.logger.error(f"Error calculating performance summary: {e}")
    
    def _record_performance(self, iteration: int, cycle_start: float, portfolio_value: float):
        """Record performance metrics for this cycle"""
        
        cycle_time = time.time() - cycle_start
        
        performance_record = {
            'iteration': iteration,
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'positions': self.current_positions.copy(),
            'signals': self.current_signals.copy(),
            'cycle_time': cycle_time
        }
        
        self.performance_history.append(performance_record)
        
        # Log basic performance
        self.logger.info(f"Portfolio Value: ${portfolio_value:,.2f} | Cycle Time: {cycle_time:.2f}s")
    
    def stop(self):
        """Stop the trading engine"""
        self.logger.info("Stopping trading engine...")
        self.is_running = False
        
        # Generate final daily return summary
        try:
            return_summary = self.execution_handler.get_daily_return_summary()
            self.logger.info(f"Final Performance Summary:")
            self.logger.info(f"  Average Daily Return: {return_summary.get('avg_daily_return', 0)*100:+.2f}%")
            self.logger.info(f"  Total Return: {return_summary.get('total_return', 0)*100:+.2f}%")
            self.logger.info(f"  Annualized Volatility: {return_summary.get('annualized_volatility', 0)*100:.2f}%")
        except Exception as e:
            self.logger.error(f"Error generating return summary: {e}")
        
        # Disconnect from IBKR
        self.execution_handler.disconnect()
        
        # Save final reports and data
        try:
            self._save_final_reports()
            
            # Generate and save final daily report
            try:
                final_daily_report = self.report_generator.generate_daily_report(self)
                if final_daily_report:
                    self.report_generator.save_daily_report(final_daily_report)
                    self.report_generator.save_daily_report_html(final_daily_report)
                    self.logger.info("Final daily report generated successfully")
            except Exception as e:
                self.logger.error(f"Error generating final daily report: {e}")
                
        except Exception as e:
            self.logger.error(f"Error saving final reports: {e}")
        
        # Friendly completion message
        print("\n" + "="*60)
        print(" 今日交易完成")
        print("="*60)
        print(" IBKR连接已断开")
        print(" 每日报告已生成")
        print(" 所有数据已保存")
        print("="*60)
        print("感谢使用量化交易引擎，期待明日再见！ ")
        print("="*60 + "\n")
        
        self.logger.info("Trading engine stopped gracefully")
    
    def _save_final_reports(self):
        """Save comprehensive final reports"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save risk metrics history
        if self.risk_manager.risk_metrics_history:
            risk_df = pd.DataFrame(self.risk_manager.risk_metrics_history)
            risk_file = os.path.join(self.config.report_dir, f'risk_metrics_{timestamp}.csv')
            risk_df.to_csv(risk_file, index=False)
            self.logger.info(f"Risk metrics saved to {risk_file}")
        
        # 2. Save performance history
        if self.performance_history:
            perf_df = pd.DataFrame(self.performance_history)
            perf_file = os.path.join(self.config.report_dir, f'performance_history_{timestamp}.csv')
            perf_df.to_csv(perf_file, index=False)
            self.logger.info(f"Performance history saved to {perf_file}")
        
        # 3. Save signal history for each symbol
        for symbol in self.config.symbols:
            if symbol in self.signal_generator.signal_history:
                signal_df = pd.DataFrame(self.signal_generator.signal_history[symbol])
                signal_file = os.path.join(self.config.report_dir, f'{symbol}_signals_{timestamp}.csv')
                signal_df.to_csv(signal_file, index=False)
                self.logger.info(f"Signal history for {symbol} saved to {signal_file}")
        
        # 4. Generate final comprehensive report
        final_report = self.generate_report()
        final_report_file = os.path.join(self.config.report_dir, f'final_trading_report_{timestamp}.json')
        self.report_generator.save_report_to_file(final_report, f'final_trading_report_{timestamp}.json')
        
        # 5. Create plots
        plot_files = self.report_generator.create_performance_plots(self)
        if plot_files:
            self.logger.info(f"Generated plots: {plot_files}")
        
        # 6. Create simple performance chart
        if self.performance_history:
            self._create_simple_performance_chart(timestamp)
    
    def _create_simple_performance_chart(self, timestamp: str):
        """Create a simple cumulative return chart"""
        
        try:
            portfolio_values = [record['portfolio_value'] for record in self.performance_history]
            timestamps = [record['timestamp'] for record in self.performance_history]
            
            if len(portfolio_values) < 2:
                return
            
            # Calculate cumulative returns
            initial_value = self.config.initial_capital
            cum_returns = [(pv / initial_value - 1) * 100 for pv in portfolio_values]
            
            if PLOTTING_AVAILABLE:
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps, cum_returns, linewidth=2, color='blue', label='Portfolio')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.title("Portfolio Cumulative Return", fontsize=14, fontweight='bold')
                plt.xlabel("Date")
                plt.ylabel("Cumulative Return (%)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                chart_file = os.path.join(self.config.report_dir, f'portfolio_chart_{timestamp}.png')
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Performance chart saved to {chart_file}")
                
            elif PLOTLY_AVAILABLE:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cum_returns,
                    mode='lines',
                    name='Portfolio Return',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red", 
                             annotation_text="Break Even")
                
                fig.update_layout(
                    title='Portfolio Cumulative Return',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    template='plotly_white',
                    height=500
                )
                
                chart_file = os.path.join(self.config.report_dir, f'portfolio_chart_{timestamp}.html')
                fig.write_html(chart_file)
                
                self.logger.info(f"Interactive chart saved to {chart_file}")
                
        except Exception as e:
            self.logger.error(f"Error creating performance chart: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive trading report"""
        return self.report_generator.generate_performance_report(self)
    
    def run_daily_cycle(self) -> Dict[str, Any]:
        """
        执行一个完整的每日交易周期：
        1. 加载数据
        2. 生成信号
        3. 组合优化
        4. 执行交易
        5. 生成日报
        
        Returns:
            Dict containing cycle results and reports
        """
        self.logger.info("开始执行每日交易周期...")
        
        cycle_results = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': 0.0,
            'daily_return': 0.0,
            'trades_executed': 0,
            'signals': {},
            'daily_report': None,
            'errors': []
        }
        
        try:
            # 1. 加载数据
            self.logger.info("1/5 - 更新市场数据...")
            current_prices = self.data_handler.get_real_time_data()
            if not current_prices:
                raise ValueError("无法获取市场数据")
            
            prices_dict = {symbol: data['price'] for symbol, data in current_prices.items()}
            
            # 更新历史数据
            self._update_historical_data_with_current_prices(current_prices)
            
            # 2. 生成交易信号
            self.logger.info("2/5 - 生成交易信号...")
            ml_signals = self.signal_generator.generate_ml_signals(self.historical_data)
            technical_signals = self.signal_generator.generate_technical_signals(self.historical_data)
            volatility_signals = self.signal_generator.generate_volatility_signals(self.historical_data)
            
            # 融合信号
            self.current_signals = self.signal_generator.fuse_signals(
                ml_signals, technical_signals, volatility_signals
            )
            cycle_results['signals'] = self.current_signals.copy()
            
            # 3. 组合优化和风险管理（含平滑换仓）
            self.logger.info("3/5 - 执行组合优化...")
            portfolio_value = sum(pos * prices_dict.get(symbol, 0) 
                                for symbol, pos in self.current_positions.items())
            cycle_results['portfolio_value'] = portfolio_value
            
            # 如果启用了平滑换仓，使用新的优化方法
            if self.config.smooth_rebalancing and hasattr(self, 'portfolio_optimizer'):
                # 准备当前权重
                current_weights = {}
                if portfolio_value > 0:
                    for symbol, position in self.current_positions.items():
                        value = position * prices_dict.get(symbol, 0)
                        current_weights[symbol] = value / portfolio_value
                
                # 准备预期收益（基于信号）
                expected_returns = {}
                for symbol, signal in self.current_signals.items():
                    # 将信号转换为预期收益预测
                    # 信号范围 [0,1]，转换为 [-5%, +5%] 的年化收益预期
                    expected_returns[symbol] = (signal - 0.5) * 0.10 / 252  # 日收益率
                
                # 准备历史收益数据用于风险估计
                risk_data = {}
                for symbol in self.current_signals.keys():
                    if symbol in self.historical_data and 'return' in self.historical_data[symbol].columns:
                        risk_data[symbol] = self.historical_data[symbol]['return'].tail(252)  # 最近一年数据
                
                # 执行带平滑换仓的组合优化
                smooth_weights, trade_sequence = self.portfolio_optimizer.optimize_with_smooth_rebalancing(
                    expected_returns=expected_returns,
                    risk_data=risk_data,
                    current_weights=current_weights,
                    current_prices=prices_dict,
                    portfolio_value=portfolio_value,
                    risk_manager=self.risk_manager
                )
                
                # 将权重转换为目标仓位
                target_positions = {}
                for symbol, weight in smooth_weights.items():
                    if weight > 0.001:  # 最小持仓阈值
                        target_value = weight * portfolio_value
                        price = prices_dict.get(symbol, 0)
                        if price > 0:
                            target_positions[symbol] = int(target_value / price)
                
                # 记录平滑换仓信息
                cycle_results['smooth_rebalancing'] = {
                    'trade_sequence': trade_sequence,
                    'target_weights': smooth_weights,
                    'trade_count': len(trade_sequence)
                }
                
                self.logger.info(f"平滑换仓: 计划执行 {len(trade_sequence)} 笔交易")
                
            else:
                # 使用传统的仓位计算方法
                target_positions = self.risk_manager.calculate_position_sizes(
                    self.current_signals, prices_dict, portfolio_value, self.current_positions
                )
            
            # 检查是否需要重新平衡
            trades_to_execute = {}
            for symbol, target_qty in target_positions.items():
                current_qty = self.current_positions.get(symbol, 0)
                if abs(target_qty - current_qty) > 0:  # 有变化就执行
                    trades_to_execute[symbol] = target_qty
            
            # 4. 执行交易
            if trades_to_execute:
                self.logger.info(f"4/5 - 执行交易: {len(trades_to_execute)} 个标的需要调整")
                execution_results = self.execution_handler.execute_trades(
                    trades_to_execute, prices_dict
                )
                
                # 更新仓位
                trades_count = 0
                for symbol, result in execution_results.items():
                    if result['status'] == 'filled':
                        self.current_positions[symbol] = trades_to_execute[symbol]
                        trades_count += 1
                        self.logger.info(f"已调整 {symbol}: {trades_to_execute[symbol]} 股")
                
                cycle_results['trades_executed'] = trades_count
            else:
                self.logger.info("4/5 - 无需执行交易，当前组合已为最优")
                cycle_results['trades_executed'] = 0
            
            # 更新每日收益
            self.execution_handler.update_daily_return(prices_dict)
            
            # 获取每日收益汇总
            return_summary = self.execution_handler.get_daily_return_summary()
            if return_summary:
                cycle_results['daily_return'] = return_summary.get('avg_daily_return', 0)
            
            # 更新风险指标
            returns_data = {symbol: df['return'] for symbol, df in self.historical_data.items() if 'return' in df.columns}
            self.risk_manager.update_risk_metrics(
                portfolio_value, self.current_positions, prices_dict, returns_data
            )
            
            # 5. 生成每日报告
            self.logger.info("5/5 - 生成每日报告...")
            daily_report = self.report_generator.generate_daily_report(self)
            if daily_report:
                # 保存报告
                self.report_generator.save_daily_report(daily_report)
                self.report_generator.save_daily_report_html(daily_report)
                cycle_results['daily_report'] = daily_report
                
                # 打印简要汇总
                self._print_daily_cycle_summary(cycle_results, daily_report)
            
            # 确保每日收益数据写入CSV（最终确认）
            self._finalize_daily_report(cycle_results, current_prices)
            
            cycle_results['success'] = True
            self.logger.info("每日交易周期执行完成！")
            
        except Exception as e:
            error_msg = f"每日交易周期执行失败: {e}"
            self.logger.error(error_msg)
            cycle_results['errors'].append(error_msg)
            cycle_results['success'] = False
            
        return cycle_results
    
    def _print_daily_cycle_summary(self, cycle_results: Dict[str, Any], daily_report: Dict[str, Any]):
        """打印每日周期执行摘要"""
        print("\n" + "="*60)
        print(" 每日交易周期执行摘要")
        print("="*60)
        print(f" 执行时间: {cycle_results['timestamp']}")
        print(f" 组合价值: ${cycle_results['portfolio_value']:,.2f}")
        print(f" 日收益率: {cycle_results['daily_return']*100:+.2f}%")
        print(f" 交易执行: {cycle_results['trades_executed']} 笔")
        
        # 显示信号状态
        signals = cycle_results.get('signals', {})
        if signals:
            print(f"\n 当前信号:")
            for symbol, signal in signals.items():
                if signal >= 0.65:
                    action = "BUY"
                elif signal <= 0.35:
                    action = "SELL"
                else:
                    action = "HOLD"
                print(f"   {symbol}: {signal:.3f} ({action})")
        
        # 报告文件信息
        if daily_report:
            date_str = daily_report.get('date', 'unknown')
            print(f"\n 报告已生成:")
            print(f"   JSON: reports/daily_report_{date_str}.json")
            print(f"   HTML: reports/daily_report_{date_str}.html")
            print(f"   CSV:  {self.execution_handler.daily_report_file}")
        
        print("="*60)
        if cycle_results['success']:
            print(" 每日周期执行成功！")
        else:
            print(" 每日周期执行失败")
            for error in cycle_results['errors']:
                print(f"   错误: {error}")
        print("="*60 + "\n")
    
    def _finalize_daily_report(self, cycle_results: Dict[str, Any], current_prices: Dict[str, float]):
        """确保每日收益数据正确写入CSV文件"""
        try:
            # 计算当前组合总价值
            total_value = 0.0
            for symbol, position in self.current_positions.items():
                if symbol in current_prices:
                    total_value += position * current_prices[symbol]
            
            # 计算每日收益率
            daily_ret = 0.0
            if self.execution_handler.last_portfolio_value is not None and self.execution_handler.last_portfolio_value > 0:
                daily_ret = (total_value - self.execution_handler.last_portfolio_value) / self.execution_handler.last_portfolio_value
            
            # 更新最后的组合价值
            self.execution_handler.last_portfolio_value = total_value
            
            # 确保报告目录存在
            os.makedirs(self.config.report_dir, exist_ok=True)
            
            # 写入CSV文件（如果不存在则创建表头）
            csv_file = self.execution_handler.daily_report_file
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, "a", encoding='utf-8') as f:
                if not file_exists:
                    f.write("date,portfolio_value,daily_return,trades_executed\n")
                
                f.write(f"{datetime.now().strftime('%Y-%m-%d')},{total_value:.2f},{daily_ret:.4%},{cycle_results.get('trades_executed', 0)}\n")
            
            print(f"今日交易完成，日报已保存到 {csv_file}")
            
        except Exception as e:
            self.logger.error(f"写入每日报告失败: {e}")
    
    def backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Run backtesting on historical data"""
        
        self.logger.info("Starting backtest...")
        
        # Load backtesting data
        if start_date or end_date:
            backtest_data = self.data_handler.load_historical_data(
                symbols=self.config.symbols,
                start_date=start_date,
                end_date=end_date
            )
        else:
            backtest_data = self.historical_data
        
        if not backtest_data:
            raise ValueError("No backtesting data available")
        
        # Initialize backtest state
        bt_positions = {symbol: 0 for symbol in self.config.symbols}
        bt_portfolio_values = []
        bt_trades = []
        bt_cash = self.config.initial_capital
        
        # Get minimum length across all symbols
        min_length = min(len(df) for df in backtest_data.values())
        start_idx = max(50, min_length // 4)  # Start after sufficient data for indicators
        
        for i in range(start_idx, min_length):
            # Create current data snapshot
            current_data = {}
            current_prices = {}
            
            for symbol, df in backtest_data.items():
                current_data[symbol] = df.iloc[:i+1].copy()
                current_prices[symbol] = float(df.iloc[i]['close'])
            
            # Generate signals
            ml_signals = self.signal_generator.generate_ml_signals(current_data)
            technical_signals = self.signal_generator.generate_technical_signals(current_data)
            volatility_signals = self.signal_generator.generate_volatility_signals(current_data)
            
            signals = self.signal_generator.fuse_signals(
                ml_signals, technical_signals, volatility_signals
            )
            
            # Calculate target positions
            portfolio_value = bt_cash + sum(pos * current_prices[symbol] 
                                          for symbol, pos in bt_positions.items())
            
            target_positions = self.risk_manager.calculate_position_sizes(
                signals, current_prices, portfolio_value, bt_positions
            )
            
            # Execute trades
            for symbol, target_qty in target_positions.items():
                current_qty = bt_positions.get(symbol, 0)
                trade_qty = target_qty - current_qty
                
                if trade_qty != 0:
                    trade_value = trade_qty * current_prices[symbol]
                    
                    # Simple execution (no transaction costs for backtest)
                    if bt_cash >= abs(trade_value) or trade_qty < 0:  # Can afford or selling
                        bt_positions[symbol] = target_qty
                        bt_cash -= trade_value
                        
                        bt_trades.append({
                            'date': backtest_data[symbol].iloc[i]['date'],
                            'symbol': symbol,
                            'quantity': trade_qty,
                            'price': current_prices[symbol],
                            'value': trade_value
                        })
            
            # Record portfolio value
            portfolio_value = bt_cash + sum(pos * current_prices[symbol] 
                                          for symbol, pos in bt_positions.items())
            bt_portfolio_values.append(portfolio_value)
        
        # Calculate backtest performance
        bt_returns = pd.Series(bt_portfolio_values).pct_change().dropna()
        bt_metrics = PerformanceMetrics.calculate_returns_metrics(bt_returns)
        
        backtest_results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.config.initial_capital,
            'final_portfolio_value': bt_portfolio_values[-1] if bt_portfolio_values else self.config.initial_capital,
            'total_return': (bt_portfolio_values[-1] / self.config.initial_capital - 1) if bt_portfolio_values else 0,
            'total_trades': len(bt_trades),
            'performance_metrics': bt_metrics,
            'portfolio_evolution': bt_portfolio_values,
            'trades': bt_trades
        }
        
        self.logger.info(f"Backtest complete. Total return: {backtest_results['total_return']:.2%}")
        
        return backtest_results


# ============================================================================
# Utility Functions and Main Execution
# ============================================================================

def load_config_from_file(config_path: str) -> TradingConfig:
    """Load configuration from file"""
    return TradingConfig.from_file(config_path)

def create_sample_config(config_path: str = 'trading_config.json'):
    """Create a sample configuration file"""
    config = TradingConfig()
    config.save_to_file(config_path)
    print(f"Sample configuration saved to {config_path}")

def run_daily_trading():
    """简化的每日交易入口函数"""
    print("="*60)
    print("启动每日量化交易系统")
    print("="*60)
    
    # 创建默认配置
    config = TradingConfig(
        simulate_trading=True,   # Paper 模式
        symbols=["AAPL", "QQQ", "VOO"],
        report_dir="reports"
    )
    
    print(f"交易标的: {config.symbols}")
    print(f"模拟交易: {config.simulate_trading}")
    print(f"报告目录: {config.report_dir}")
    print("="*60)
    
    try:
        # 初始化交易引擎
        engine = QuantTradingEnginePlus(config)
        engine.initialize()
        
        # 执行每日交易周期
        results = engine.run_daily_cycle()
        
        # 额外调用简化日报保存方法
        try:
            current_prices = engine.data_handler.get_real_time_data(config.symbols)
            if current_prices:
                price_dict = {symbol: data['price'] for symbol, data in current_prices.items()}
                engine.execution_handler.save_daily_report(price_dict)
        except Exception as e:
            print(f"保存额外日报时出错: {e}")
        
        if results['success']:
            print("\n 每日交易周期执行成功!")
        else:
            print("\n 每日交易周期执行失败")
            for error in results.get('errors', []):
                print(f"   错误: {error}")
                
    except KeyboardInterrupt:
        print("\n⏸  用户中断操作")
    except Exception as e:
        print(f"\n 执行出错: {e}")
    finally:
        print("\n" + "="*50)
        print("系统自动断开连接，可以休息啦～")
        print(" 所有交易数据已保存")
        print(" 日报文件: reports/daily_report.csv")
        print("="*50)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quant Trading Engine Plus')
    parser.add_argument('--config', type=str, default='trading_config.json', 
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['trade', 'backtest', 'config'], 
                       default='trade', help='Execution mode')
    parser.add_argument('--symbols', nargs='+', default=['QQQ', 'VOO', 'AAPL'], 
                       help='Trading symbols')
    parser.add_argument('--runtime', type=int, default=0, 
                       help='Runtime in seconds (0 = infinite)')
    parser.add_argument('--iterations', type=int, default=0, 
                       help='Maximum iterations (0 = infinite)')
    
    args = parser.parse_args()
    
    if args.mode == 'config':
        create_sample_config(args.config)
        return
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config_from_file(args.config)
    else:
        config = TradingConfig()
        config.symbols = args.symbols
    
    # Override symbols if provided
    if args.symbols != ['QQQ', 'VOO', 'AAPL']:
        config.symbols = args.symbols
    
    print("="*60)
    print("Quant Trading Engine Plus")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Symbols: {config.symbols}")
    print(f"Simulation: {config.simulate_trading}")
    print("="*60)
    
    try:
        # Initialize trading engine
        engine = QuantTradingEnginePlus(config)
        engine.initialize()
        
        if args.mode == 'trade':
            # Run live/simulation trading
            engine.run_strategy(
                max_iterations=args.iterations if args.iterations > 0 else None,
                run_time_seconds=args.runtime if args.runtime > 0 else None
            )
        elif args.mode == 'backtest':
            # Run backtest
            results = engine.backtest()
            print("\n" + "="*40)
            print("BACKTEST RESULTS")
            print("="*40)
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
            print(f"Total Trades: {results['total_trades']}")
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
            print("="*40)
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
    finally:
        print("\n" + "="*50)
        print(" 程序执行完成")
        print(" 所有报告已生成")
        print(" 交易数据已保存")
        print("="*50)
        print("谢谢使用！期待下次交易 ")
        print("="*50)

if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys
    
    if len(sys.argv) == 1:
        # 没有命令行参数，运行简化的每日交易
        run_daily_trading()
    else:
        # 有命令行参数，运行完整的主程序
        main()
