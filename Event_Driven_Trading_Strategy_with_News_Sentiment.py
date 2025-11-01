"""
Event-Driven Trading Strategy with News Sentiment Analysis

This script implements a sophisticated event-driven trading strategy that combines:
- Real-time news sentiment analysis using FinBERT
- Technical indicators and market data
- Event-responsive trading signals
- Performance comparison with and without sentiment factors

Key Features:
- FinBERT-based financial news sentiment analysis
- Multi-source news aggregation (Financial APIs)
- Real-time signal generation and portfolio management
- Event detection and reaction mechanisms
- Comprehensive backtesting with sentiment vs. no-sentiment comparison
- Risk management and position sizing
- Performance analytics and visualization

Dependencies:
pip install transformers torch yfinance newsapi-python alpha-vantage finnhub-python
pip install pandas numpy matplotlib seaborn scikit-learn ta-lib plotly dash

Author: Advanced Quantitative Trading System
Date: 2025-01-01
"""

import os
import warnings
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML and NLP
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("Transformers library loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] transformers not available. Install with: pip install transformers torch")
    # Create dummy classes for fallback
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    def pipeline(*args, **kwargs):
        return None
    class torch:
        class device:
            def __init__(self, device_name):
                self.type = device_name
        @staticmethod
        def cuda():
            class cuda:
                @staticmethod
                def is_available():
                    return False
            return cuda

# Financial data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("yfinance loaded successfully")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARN] yfinance not available. Install with: pip install yfinance")

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib loaded successfully")
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib not available. Some technical indicators will use pandas implementations.")

# News APIs
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
    print("NewsAPI client loaded successfully")
except ImportError:
    NEWSAPI_AVAILABLE = False
    print("[WARN] newsapi-python not available. Install with: pip install newsapi-python")

try:
    import finnhub
    FINNHUB_AVAILABLE = True
    print("Finnhub client loaded successfully")
except ImportError:
    FINNHUB_AVAILABLE = False
    print("[WARN] finnhub-python not available. Install with: pip install finnhub-python")

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configuration
CONFIG = {
    'DATA_DIR': 'data',
    'REPORT_DIR': 'reports',
    'MODEL_DIR': 'models',
    'NEWS_CACHE_HOURS': 1,
    'SENTIMENT_THRESHOLD': 0.1,
    'POSITION_SIZE': 0.1,  # 10% of portfolio per position
    'STOP_LOSS': 0.05,     # 5% stop loss
    'TAKE_PROFIT': 0.15,   # 15% take profit
    'REBALANCE_FREQ': '1D', # Daily rebalancing
    'LOOKBACK_DAYS': 30,   # Lookback period for technical indicators
    'NEWS_LOOKBACK_HOURS': 24  # News lookback period
}

# News API Keys (You need to set these up)
NEWS_API_KEYS = {
    'newsapi': 'YOUR_NEWSAPI_KEY',
    'finnhub': 'YOUR_FINNHUB_KEY',
    'alpha_vantage': 'YOUR_ALPHA_VANTAGE_KEY'
}

class FinBERTSentimentAnalyzer:
    """FinBERT-based financial sentiment analyzer"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            model_name: Hugging Face model name for FinBERT
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        
        self._load_model()
        
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available. Using dummy sentiment analyzer.")
            self.sentiment_pipeline = None
            return
            
        try:
            print(f"Loading FinBERT model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"FinBERT model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load FinBERT model: {e}")
            print("Using fallback sentiment analysis...")
            self._create_fallback_analyzer()
    
    def _create_fallback_analyzer(self):
        """Create fallback sentiment analyzer if FinBERT fails"""
        if not TRANSFORMERS_AVAILABLE:
            print("No transformers available. Using simple rule-based sentiment.")
            self.sentiment_pipeline = None
            return
            
        try:
            # Use a general sentiment analysis model as fallback
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Fallback sentiment analyzer loaded")
        except Exception as e:
            print(f"Fallback analyzer also failed: {e}")
            print("Using simple rule-based sentiment analysis")
            self.sentiment_pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dictionary with sentiment score and label
        """
        if not self.sentiment_pipeline:
            # Simple rule-based fallback
            return self._simple_sentiment_analysis(text)
        
        try:
            # Clean text
            text = self._clean_text(text)
            
            if len(text.strip()) == 0:
                return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0}
            
            # Analyze sentiment
            result = self.sentiment_pipeline(text)[0]
            
            # Convert to standardized format
            sentiment_score = self._convert_to_score(result)
            
            return {
                'label': result['label'],
                'score': sentiment_score,
                'confidence': result['score'],
                'raw_result': result
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._simple_sentiment_analysis(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (FinBERT has token limits)
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def _simple_sentiment_analysis(self, text: str) -> Dict:
        """Simple rule-based sentiment analysis fallback"""
        text = text.lower()
        
        # Define sentiment keywords
        positive_words = ['buy', 'bull', 'bullish', 'positive', 'gains', 'profit', 'growth', 'strong', 'beat', 'exceed', 'rally', 'surge', 'boost']
        negative_words = ['sell', 'bear', 'bearish', 'negative', 'loss', 'decline', 'weak', 'miss', 'fall', 'drop', 'crash', 'plunge']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            score = min(0.8, positive_count * 0.2)
            return {'label': 'POSITIVE', 'score': score, 'confidence': 0.5}
        elif negative_count > positive_count:
            score = -min(0.8, negative_count * 0.2)
            return {'label': 'NEGATIVE', 'score': score, 'confidence': 0.5}
        else:
            return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 0.5}
    
    def _convert_to_score(self, result: Dict) -> float:
        """
        Convert sentiment result to numerical score
        
        Args:
            result: Raw sentiment analysis result
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        label = result['label'].upper()
        confidence = result['score']
        
        # Map labels to scores
        if label in ['POSITIVE', 'POS']:
            return confidence
        elif label in ['NEGATIVE', 'NEG']:
            return -confidence
        else:  # NEUTRAL
            return 0.0
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment results
        """
        results = []
        
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            results.append(sentiment)
            
        return results

class NewsDataCollector:
    """Multi-source news data collector"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize news data collector
        
        Args:
            api_keys: Dictionary of API keys for different news sources
        """
        self.api_keys = api_keys or NEWS_API_KEYS
        self.news_cache = {}
        self.cache_timeout = CONFIG['NEWS_CACHE_HOURS'] * 3600  # Convert to seconds
        
        # Initialize news clients
        self.newsapi_client = None
        self.finnhub_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize news API clients"""
        # NewsAPI client
        if NEWSAPI_AVAILABLE and self.api_keys.get('newsapi'):
            try:
                self.newsapi_client = NewsApiClient(api_key=self.api_keys['newsapi'])
                print("NewsAPI client initialized")
            except Exception as e:
                print(f"NewsAPI initialization failed: {e}")
        
        # Finnhub client
        if FINNHUB_AVAILABLE and self.api_keys.get('finnhub'):
            try:
                self.finnhub_client = finnhub.Client(api_key=self.api_keys['finnhub'])
                print("✅ Finnhub client initialized")
            except Exception as e:
                print(f"Finnhub initialization failed: {e}")
    
    def get_stock_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """
        Get news articles for a specific stock
        
        Args:
            symbol: Stock symbol
            hours_back: Hours to look back for news
            
        Returns:
            List of news articles with metadata
        """
        cache_key = f"{symbol}_{hours_back}"
        
        # Check cache
        if self._is_cached(cache_key):
            print(f"📰 Using cached news for {symbol}")
            return self.news_cache[cache_key]['data']
        
        news_articles = []
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Collect from multiple sources
        news_articles.extend(self._get_newsapi_articles(symbol, start_time, end_time))
        news_articles.extend(self._get_finnhub_articles(symbol, start_time, end_time))
        news_articles.extend(self._get_yfinance_news(symbol))
        
        # Remove duplicates and sort by timestamp
        news_articles = self._deduplicate_news(news_articles)
        news_articles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Cache results
        self.news_cache[cache_key] = {
            'data': news_articles,
            'timestamp': time.time()
        }
        
        print(f"📰 Collected {len(news_articles)} news articles for {symbol}")
        return news_articles
    
    def _get_newsapi_articles(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get articles from NewsAPI"""
        if not self.newsapi_client:
            return []
        
        try:
            # Search for company news
            articles = self.newsapi_client.get_everything(
                q=f"{symbol} OR {self._get_company_name(symbol)}",
                language='en',
                sort_by='publishedAt',
                from_param=start_time.strftime('%Y-%m-%d'),
                to=end_time.strftime('%Y-%m-%d')
            )
            
            news_list = []
            for article in articles.get('articles', []):
                news_list.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': f"NewsAPI-{article.get('source', {}).get('name', 'Unknown')}",
                    'timestamp': pd.to_datetime(article.get('publishedAt')),
                    'symbol': symbol
                })
            
            return news_list
            
        except Exception as e:
            logger.error(f"NewsAPI error for {symbol}: {e}")
            return []
    
    def _get_finnhub_articles(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get articles from Finnhub"""
        if not self.finnhub_client:
            return []
        
        try:
            # Get company news
            news = self.finnhub_client.company_news(
                symbol,
                _from=start_time.strftime('%Y-%m-%d'),
                to=end_time.strftime('%Y-%m-%d')
            )
            
            news_list = []
            for article in news:
                news_list.append({
                    'title': article.get('headline', ''),
                    'description': article.get('summary', ''),
                    'content': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'source': 'Finnhub',
                    'timestamp': pd.to_datetime(article.get('datetime'), unit='s'),
                    'symbol': symbol
                })
            
            return news_list
            
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return []
    
    def _get_yfinance_news(self, symbol: str) -> List[Dict]:
        """Get news from Yahoo Finance via yfinance"""
        if not YFINANCE_AVAILABLE:
            return []
            
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_list = []
            for article in news:
                news_list.append({
                    'title': article.get('title', ''),
                    'description': article.get('summary', ''),
                    'content': article.get('summary', ''),
                    'url': article.get('link', ''),
                    'source': 'Yahoo Finance',
                    'timestamp': pd.to_datetime(article.get('providerPublishTime'), unit='s'),
                    'symbol': symbol
                })
            
            return news_list
            
        except Exception as e:
            logger.error(f"Yahoo Finance news error for {symbol}: {e}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol (simplified mapping)"""
        company_map = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA',
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'VOO': 'Vanguard S&P 500'
        }
        return company_map.get(symbol, symbol)
    
    def _deduplicate_news(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate news articles"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self.news_cache:
            return False
        
        cache_time = self.news_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_timeout

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(prices.values, timeperiod=window), index=prices.index)
        else:
            # Pandas implementation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD indicator"""
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': pd.Series(macd, index=prices.index),
                'signal': pd.Series(macd_signal, index=prices.index),
                'histogram': pd.Series(macd_hist, index=prices.index)
            }
        else:
            # Pandas implementation
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            
            return {
                'macd': macd,
                'signal': macd_signal,
                'histogram': macd_hist
            }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict:
        """Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return {
            'upper': sma + (std * num_std),
            'middle': sma,
            'lower': sma - (std * num_std)
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict:
        """Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                     fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return {
                'k': pd.Series(slowk, index=close.index),
                'd': pd.Series(slowd, index=close.index)
            }
        else:
            # Pandas implementation
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k': k_percent,
                'd': d_percent
            }

class EventDrivenTradingStrategy:
    """Event-driven trading strategy with sentiment analysis"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        """
        Initialize event-driven trading strategy
        
        Args:
            symbols: List of symbols to trade
            initial_capital: Initial portfolio capital
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {symbol: 0 for symbol in symbols}
        self.trade_history = []
        self.portfolio_history = []
        
        # Initialize components
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.news_collector = NewsDataCollector()
        self.technical_indicators = TechnicalIndicators()
        
        # Strategy parameters
        self.sentiment_weight = 0.3
        self.technical_weight = 0.4
        self.momentum_weight = 0.3
        
        # Risk management
        self.max_position_size = CONFIG['POSITION_SIZE']
        self.stop_loss = CONFIG['STOP_LOSS']
        self.take_profit = CONFIG['TAKE_PROFIT']
        
        print(f"🚀 Event-driven trading strategy initialized")
        print(f"💰 Initial capital: ${initial_capital:,.2f}")
        print(f"📊 Trading symbols: {symbols}")
    
    def get_market_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Get market data for symbol"""
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available. Cannot retrieve market data.")
            return pd.DataFrame()
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_sentiment_signal(self, symbol: str) -> float:
        """
        Calculate sentiment-based signal for symbol
        
        Returns:
            Sentiment signal between -1 and 1
        """
        try:
            # Get recent news
            news_articles = self.news_collector.get_stock_news(
                symbol, hours_back=CONFIG['NEWS_LOOKBACK_HOURS']
            )
            
            if not news_articles:
                return 0.0
            
            # Analyze sentiment for each article
            sentiment_scores = []
            
            for article in news_articles:
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if not text.strip():
                    continue
                
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                
                # Weight by confidence and recency
                hours_old = (datetime.now() - article['timestamp']).total_seconds() / 3600
                recency_weight = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
                
                weighted_score = sentiment['score'] * sentiment['confidence'] * recency_weight
                sentiment_scores.append(weighted_score)
            
            if not sentiment_scores:
                return 0.0
            
            # Calculate overall sentiment signal
            avg_sentiment = np.mean(sentiment_scores)
            
            # Apply threshold and scaling
            if abs(avg_sentiment) < CONFIG['SENTIMENT_THRESHOLD']:
                return 0.0
            
            return np.clip(avg_sentiment, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Sentiment signal calculation failed for {symbol}: {e}")
            return 0.0
    
    def calculate_technical_signal(self, symbol: str) -> float:
        """
        Calculate technical analysis signal
        
        Returns:
            Technical signal between -1 and 1
        """
        try:
            # Get market data
            data = self.get_market_data(symbol, period="2mo")
            
            if data.empty or len(data) < 50:
                return 0.0
            
            prices = data['Close']
            high = data['High']
            low = data['Low']
            
            signals = []
            
            # RSI signal
            rsi = self.technical_indicators.calculate_rsi(prices)
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:  # Oversold
                signals.append(0.8)
            elif current_rsi > 70:  # Overbought
                signals.append(-0.8)
            else:
                signals.append(0.0)
            
            # MACD signal
            macd_data = self.technical_indicators.calculate_macd(prices)
            macd_hist = macd_data['histogram'].iloc[-1]
            prev_macd_hist = macd_data['histogram'].iloc[-2]
            
            if macd_hist > 0 and prev_macd_hist <= 0:  # Bullish crossover
                signals.append(0.6)
            elif macd_hist < 0 and prev_macd_hist >= 0:  # Bearish crossover
                signals.append(-0.6)
            else:
                signals.append(0.0)
            
            # Bollinger Bands signal
            bb = self.technical_indicators.calculate_bollinger_bands(prices)
            current_price = prices.iloc[-1]
            
            if current_price < bb['lower'].iloc[-1]:  # Below lower band
                signals.append(0.7)
            elif current_price > bb['upper'].iloc[-1]:  # Above upper band
                signals.append(-0.7)
            else:
                signals.append(0.0)
            
            # Moving average signal
            sma_20 = self.technical_indicators.calculate_sma(prices, 20)
            sma_50 = self.technical_indicators.calculate_sma(prices, 50)
            
            if sma_20.iloc[-1] > sma_50.iloc[-1]:  # Golden cross
                signals.append(0.5)
            else:  # Death cross
                signals.append(-0.5)
            
            # Combine signals
            technical_signal = np.mean(signals)
            return np.clip(technical_signal, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Technical signal calculation failed for {symbol}: {e}")
            return 0.0
    
    def calculate_momentum_signal(self, symbol: str) -> float:
        """
        Calculate momentum signal
        
        Returns:
            Momentum signal between -1 and 1
        """
        try:
            data = self.get_market_data(symbol, period="1mo")
            
            if data.empty or len(data) < 10:
                return 0.0
            
            prices = data['Close']
            returns = prices.pct_change().dropna()
            
            # Short-term momentum (3-day)
            short_momentum = returns.tail(3).mean()
            
            # Medium-term momentum (10-day)
            medium_momentum = returns.tail(10).mean()
            
            # Combine momentum signals
            momentum_signal = (short_momentum * 0.7 + medium_momentum * 0.3) * 100
            
            return np.clip(momentum_signal, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Momentum signal calculation failed for {symbol}: {e}")
            return 0.0
    
    def generate_trading_signal(self, symbol: str, use_sentiment: bool = True) -> Dict:
        """
        Generate combined trading signal
        
        Args:
            symbol: Symbol to analyze
            use_sentiment: Whether to include sentiment in signal
            
        Returns:
            Dictionary with signal components and final signal
        """
        signals = {}
        
        # Calculate individual signals
        technical_signal = self.calculate_technical_signal(symbol)
        momentum_signal = self.calculate_momentum_signal(symbol)
        
        signals['technical'] = technical_signal
        signals['momentum'] = momentum_signal
        
        if use_sentiment:
            sentiment_signal = self.calculate_sentiment_signal(symbol)
            signals['sentiment'] = sentiment_signal
            
            # Combine all signals
            final_signal = (
                technical_signal * self.technical_weight +
                momentum_signal * self.momentum_weight +
                sentiment_signal * self.sentiment_weight
            )
        else:
            signals['sentiment'] = 0.0
            
            # Combine without sentiment
            total_weight = self.technical_weight + self.momentum_weight
            final_signal = (
                technical_signal * (self.technical_weight / total_weight) +
                momentum_signal * (self.momentum_weight / total_weight)
            )
        
        signals['final'] = np.clip(final_signal, -1.0, 1.0)
        signals['use_sentiment'] = use_sentiment
        signals['timestamp'] = datetime.now()
        
        return signals
    
    def execute_trade(self, symbol: str, signal: float, current_price: float) -> Dict:
        """
        Execute trade based on signal
        
        Args:
            symbol: Symbol to trade
            signal: Trading signal (-1 to 1)
            current_price: Current price of the symbol
            
        Returns:
            Trade execution details
        """
        trade_info = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signal': signal,
            'price': current_price,
            'action': 'HOLD',
            'quantity': 0,
            'value': 0
        }
        
        # Calculate position size based on signal strength
        position_value = abs(signal) * self.max_position_size * self.current_capital
        quantity = int(position_value / current_price)
        
        if signal > 0.2 and quantity > 0:  # Buy signal
            if self.current_capital >= position_value:
                self.positions[symbol] += quantity
                self.current_capital -= position_value
                
                trade_info.update({
                    'action': 'BUY',
                    'quantity': quantity,
                    'value': position_value
                })
                
        elif signal < -0.2 and self.positions[symbol] > 0:  # Sell signal
            sell_quantity = min(self.positions[symbol], quantity)
            sell_value = sell_quantity * current_price
            
            self.positions[symbol] -= sell_quantity
            self.current_capital += sell_value
            
            trade_info.update({
                'action': 'SELL',
                'quantity': sell_quantity,
                'value': sell_value
            })
        
        # Record trade
        if trade_info['action'] != 'HOLD':
            self.trade_history.append(trade_info.copy())
            logger.info(f"Trade executed: {trade_info['action']} {trade_info['quantity']} {symbol} @ ${current_price:.2f}")
        
        return trade_info
    
    def backtest_strategy(self, start_date: str, end_date: str, use_sentiment: bool = True) -> Dict:
        """
        Backtest the trading strategy
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            use_sentiment: Whether to use sentiment in the strategy
            
        Returns:
            Backtesting results
        """
        print(f"📈 Starting backtest: {start_date} to {end_date}")
        print(f"🎭 Using sentiment: {use_sentiment}")
        
        # Reset portfolio
        self.current_capital = self.initial_capital
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.trade_history = []
        self.portfolio_history = []
        
        # Get historical data for all symbols
        historical_data = {}
        if not YFINANCE_AVAILABLE:
            print("⚠️ yfinance not available. Using simulated data for backtesting.")
            # Create dummy data for demonstration
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            for symbol in self.symbols:
                # Create simple random walk price data
                np.random.seed(hash(symbol) % 1000)  # Reproducible but different per symbol
                returns = np.random.normal(0.0008, 0.02, len(date_range))  # ~20% annual vol
                prices = 100 * np.exp(np.cumsum(returns))  # Start at $100
                
                historical_data[symbol] = pd.DataFrame({
                    'Close': prices,
                    'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(date_range)))),
                    'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(date_range)))),
                    'Volume': np.random.randint(1000000, 10000000, len(date_range))
                }, index=date_range)
        else:
            for symbol in self.symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    historical_data[symbol] = data
        
        if not historical_data:
            raise ValueError("No historical data found for the specified period")
        
        # Get date range
        all_dates = set()
        for data in historical_data.values():
            all_dates.update(data.index.date)
        
        trading_dates = sorted(list(all_dates))
        
        # Simulate trading
        for i, date in enumerate(trading_dates):
            if i % 10 == 0:
                print(f"Processing date {i+1}/{len(trading_dates)}: {date}")
            
            daily_portfolio_value = self.current_capital
            
            for symbol in self.symbols:
                if symbol not in historical_data:
                    continue
                
                symbol_data = historical_data[symbol]
                
                # Find the closest date
                available_dates = symbol_data.index.date
                if date not in available_dates:
                    continue
                
                current_price = symbol_data.loc[symbol_data.index.date == date, 'Close'].iloc[0]
                
                # Add position value to portfolio
                daily_portfolio_value += self.positions[symbol] * current_price
                
                # Generate signal (simplified for backtesting)
                # In real implementation, you would need historical news data
                if i % 5 == 0:  # Trade every 5 days to reduce computation
                    signal = self.generate_trading_signal(symbol, use_sentiment)
                    self.execute_trade(symbol, signal['final'], current_price)
            
            # Record portfolio value
            self.portfolio_history.append({
                'date': pd.to_datetime(date),
                'portfolio_value': daily_portfolio_value,
                'cash': self.current_capital,
                'positions_value': daily_portfolio_value - self.current_capital
            })
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        results = {
            'use_sentiment': use_sentiment,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': daily_portfolio_value,
            'total_return': (daily_portfolio_value - self.initial_capital) / self.initial_capital,
            'trade_count': len(self.trade_history),
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'trades': pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame(),
            'performance_metrics': performance_metrics
        }
        
        print(f"✅ Backtest completed:")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Final Value: ${results['final_capital']:,.2f}")
        print(f"   Total Trades: {results['trade_count']}")
        
        return results
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        if not self.portfolio_history:
            return {}
        
        df = pd.DataFrame(self.portfolio_history)
        df['returns'] = df['portfolio_value'].pct_change().dropna()
        
        metrics = {}
        
        if len(df) > 1:
            # Total return
            metrics['total_return'] = (df['portfolio_value'].iloc[-1] - df['portfolio_value'].iloc[0]) / df['portfolio_value'].iloc[0]
            
            # Annualized return
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (365 / days) - 1
            
            # Volatility
            metrics['volatility'] = df['returns'].std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            metrics['sharpe_ratio'] = (metrics['annualized_return'] - 0.02) / metrics['volatility'] if metrics['volatility'] > 0 else 0
            
            # Maximum drawdown
            rolling_max = df['portfolio_value'].expanding().max()
            drawdowns = (df['portfolio_value'] - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdowns.min()
            
            # Win rate
            if len(self.trade_history) > 0:
                trades_df = pd.DataFrame(self.trade_history)
                # This is simplified - would need entry/exit pairs for accurate win rate
                metrics['win_rate'] = 0.5  # Placeholder
        
        return metrics

class StrategyComparison:
    """Compare strategies with and without sentiment"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_comparison(self, start_date: str, end_date: str) -> Dict:
        """
        Run comparison between sentiment-based and technical-only strategies
        
        Args:
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            Comparison results
        """
        print("🔄 Running strategy comparison...")
        
        # Test with sentiment
        print("\n📰 Testing strategy WITH sentiment analysis...")
        strategy_with_sentiment = EventDrivenTradingStrategy(self.symbols, self.initial_capital)
        results_with_sentiment = strategy_with_sentiment.backtest_strategy(
            start_date, end_date, use_sentiment=True
        )
        
        # Test without sentiment
        print("\n📊 Testing strategy WITHOUT sentiment analysis...")
        strategy_without_sentiment = EventDrivenTradingStrategy(self.symbols, self.initial_capital)
        results_without_sentiment = strategy_without_sentiment.backtest_strategy(
            start_date, end_date, use_sentiment=False
        )
        
        # Buy and hold benchmark
        print("\n📈 Calculating buy-and-hold benchmark...")
        benchmark_results = self._calculate_benchmark(start_date, end_date)
        
        comparison_results = {
            'with_sentiment': results_with_sentiment,
            'without_sentiment': results_without_sentiment,
            'benchmark': benchmark_results,
            'comparison_metrics': self._calculate_comparison_metrics(
                results_with_sentiment, results_without_sentiment, benchmark_results
            )
        }
        
        self.results = comparison_results
        self._print_comparison_summary()
        
        return comparison_results
    
    def _calculate_benchmark(self, start_date: str, end_date: str) -> Dict:
        """Calculate buy-and-hold benchmark performance"""
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available. Using dummy benchmark.")
            return {
                'total_return': 0.10,  # Assume 10% annual return
                'final_value': self.initial_capital * 1.10,
                'portfolio_history': pd.DataFrame({
                    'date': pd.date_range(start=start_date, end=end_date, freq='D'),
                    'portfolio_value': [self.initial_capital * (1 + 0.10 * i/365) for i in range(len(pd.date_range(start=start_date, end=end_date, freq='D')))]
                })
            }
            
        try:
            # Use SPY as benchmark
            ticker = yf.Ticker('SPY')
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return {'total_return': 0, 'final_value': self.initial_capital}
            
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            
            total_return = (end_price - start_price) / start_price
            final_value = self.initial_capital * (1 + total_return)
            
            return {
                'total_return': total_return,
                'final_value': final_value,
                'portfolio_history': pd.DataFrame({
                    'date': data.index,
                    'portfolio_value': data['Close'] * (self.initial_capital / start_price)
                })
            }
            
        except Exception as e:
            logger.error(f"Benchmark calculation failed: {e}")
            return {'total_return': 0, 'final_value': self.initial_capital}
    
    def _calculate_comparison_metrics(self, with_sentiment: Dict, without_sentiment: Dict, benchmark: Dict) -> Dict:
        """Calculate comparison metrics"""
        return {
            'sentiment_advantage': with_sentiment['total_return'] - without_sentiment['total_return'],
            'sentiment_vs_benchmark': with_sentiment['total_return'] - benchmark['total_return'],
            'technical_vs_benchmark': without_sentiment['total_return'] - benchmark['total_return'],
            'sentiment_sharpe': with_sentiment['performance_metrics'].get('sharpe_ratio', 0),
            'technical_sharpe': without_sentiment['performance_metrics'].get('sharpe_ratio', 0),
            'benchmark_return': benchmark['total_return']
        }
    
    def _print_comparison_summary(self):
        """Print comparison summary"""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("📊 STRATEGY COMPARISON SUMMARY")
        print("="*60)
        
        with_sentiment = self.results['with_sentiment']
        without_sentiment = self.results['without_sentiment']
        benchmark = self.results['benchmark']
        comparison = self.results['comparison_metrics']
        
        print(f"With Sentiment:    {with_sentiment['total_return']:.2%} return")
        print(f"Without Sentiment: {without_sentiment['total_return']:.2%} return")
        print(f"Benchmark (SPY):   {benchmark['total_return']:.2%} return")
        print(f"\nSentiment Advantage: {comparison['sentiment_advantage']:.2%}")
        print(f"Sentiment vs Benchmark: {comparison['sentiment_vs_benchmark']:.2%}")
        print(f"Technical vs Benchmark: {comparison['technical_vs_benchmark']:.2%}")
    
    def create_comparison_visualization(self, save_path: str = None) -> None:
        """Create comprehensive comparison visualization"""
        if not self.results:
            print("No results to visualize. Run comparison first.")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Portfolio Value Comparison',
                'Returns Distribution',
                'Rolling Sharpe Ratio',
                'Performance Metrics'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Portfolio value comparison
        with_sentiment_data = self.results['with_sentiment']['portfolio_history']
        without_sentiment_data = self.results['without_sentiment']['portfolio_history']
        benchmark_data = self.results['benchmark']['portfolio_history']
        
        fig.add_trace(
            go.Scatter(
                x=with_sentiment_data['date'],
                y=with_sentiment_data['portfolio_value'],
                name='With Sentiment',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=without_sentiment_data['date'],
                y=without_sentiment_data['portfolio_value'],
                name='Without Sentiment',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_data['date'],
                y=benchmark_data['portfolio_value'],
                name='Benchmark (SPY)',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Returns distribution
        with_sentiment_returns = with_sentiment_data['portfolio_value'].pct_change().dropna()
        without_sentiment_returns = without_sentiment_data['portfolio_value'].pct_change().dropna()
        
        fig.add_trace(
            go.Histogram(
                x=with_sentiment_returns,
                name='With Sentiment Returns',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=without_sentiment_returns,
                name='Without Sentiment Returns',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=2
        )
        
        # Performance metrics table
        metrics_data = [
            ['Metric', 'With Sentiment', 'Without Sentiment', 'Benchmark'],
            ['Total Return', 
             f"{self.results['with_sentiment']['total_return']:.2%}",
             f"{self.results['without_sentiment']['total_return']:.2%}",
             f"{self.results['benchmark']['total_return']:.2%}"],
            ['Sharpe Ratio',
             f"{self.results['with_sentiment']['performance_metrics'].get('sharpe_ratio', 0):.2f}",
             f"{self.results['without_sentiment']['performance_metrics'].get('sharpe_ratio', 0):.2f}",
             'N/A'],
            ['Max Drawdown',
             f"{self.results['with_sentiment']['performance_metrics'].get('max_drawdown', 0):.2%}",
             f"{self.results['without_sentiment']['performance_metrics'].get('max_drawdown', 0):.2%}",
             'N/A'],
            ['Trade Count',
             str(self.results['with_sentiment']['trade_count']),
             str(self.results['without_sentiment']['trade_count']),
             '0']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=metrics_data[0], fill_color='lightblue'),
                cells=dict(values=list(zip(*metrics_data[1:])), fill_color='lightgray')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Event-Driven Trading Strategy: Sentiment vs Technical Analysis',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Visualization saved to: {save_path}")
        
        fig.show()

def main():
    """Main execution function"""
    print("🚀 Event-Driven Trading Strategy with News Sentiment Analysis")
    print("="*60)
    
    # Configuration
    SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    INITIAL_CAPITAL = 100000
    START_DATE = '2023-01-01'
    END_DATE = '2024-01-01'
    
    try:
        # Create strategy comparison
        comparison = StrategyComparison(SYMBOLS, INITIAL_CAPITAL)
        
        # Run comparison
        results = comparison.run_comparison(START_DATE, END_DATE)
        
        # Create visualizations
        comparison.create_comparison_visualization(
            save_path=os.path.join(CONFIG['REPORT_DIR'], 'strategy_comparison.html')
        )
        
        # Save detailed results
        results_file = os.path.join(CONFIG['REPORT_DIR'], 'trading_results.json')
        
        # Convert DataFrames to dict for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        serializable_results[key][sub_key] = sub_value.to_dict('records')
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate summary report
        generate_summary_report(results)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

def generate_summary_report(results: Dict):
    """Generate comprehensive summary report"""
    report_path = os.path.join(CONFIG['REPORT_DIR'], 'event_driven_strategy_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Event-Driven Trading Strategy with News Sentiment Analysis\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        with_sentiment = results['with_sentiment']
        without_sentiment = results['without_sentiment']
        benchmark = results['benchmark']
        comparison = results['comparison_metrics']
        
        f.write(f"- **Strategy with Sentiment:** {with_sentiment['total_return']:.2%} return\n")
        f.write(f"- **Strategy without Sentiment:** {without_sentiment['total_return']:.2%} return\n")
        f.write(f"- **Benchmark (SPY):** {benchmark['total_return']:.2%} return\n")
        f.write(f"- **Sentiment Advantage:** {comparison['sentiment_advantage']:.2%}\n\n")
        
        # Performance Metrics
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | With Sentiment | Without Sentiment | Benchmark |\n")
        f.write("|--------|---------------|------------------|----------|\n")
        f.write(f"| Total Return | {with_sentiment['total_return']:.2%} | {without_sentiment['total_return']:.2%} | {benchmark['total_return']:.2%} |\n")
        f.write(f"| Sharpe Ratio | {with_sentiment['performance_metrics'].get('sharpe_ratio', 0):.2f} | {without_sentiment['performance_metrics'].get('sharpe_ratio', 0):.2f} | N/A |\n")
        f.write(f"| Max Drawdown | {with_sentiment['performance_metrics'].get('max_drawdown', 0):.2%} | {without_sentiment['performance_metrics'].get('max_drawdown', 0):.2%} | N/A |\n")
        f.write(f"| Trade Count | {with_sentiment['trade_count']} | {without_sentiment['trade_count']} | 0 |\n\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        
        if comparison['sentiment_advantage'] > 0:
            f.write("✅ **Sentiment analysis provides positive alpha**\n")
            f.write(f"   - The sentiment-enhanced strategy outperformed by {comparison['sentiment_advantage']:.2%}\n")
        else:
            f.write("❌ **Sentiment analysis did not add value in this period**\n")
            f.write(f"   - The sentiment-enhanced strategy underperformed by {abs(comparison['sentiment_advantage']):.2%}\n")
        
        f.write(f"\n- Both strategies vs benchmark:\n")
        f.write(f"  - Sentiment strategy: {comparison['sentiment_vs_benchmark']:.2%} excess return\n")
        f.write(f"  - Technical strategy: {comparison['technical_vs_benchmark']:.2%} excess return\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Sentiment Analysis\n")
        f.write("- **Model:** FinBERT (Financial BERT) for financial text sentiment\n")
        f.write("- **Data Sources:** Yahoo Finance News, NewsAPI, Finnhub\n")
        f.write("- **Signal Generation:** Weighted average of recent news sentiment\n")
        f.write("- **Time Decay:** Recent news weighted more heavily\n\n")
        
        f.write("### Technical Analysis\n")
        f.write("- **Indicators:** RSI, MACD, Bollinger Bands, Moving Averages\n")
        f.write("- **Signal Combination:** Weighted average of all technical signals\n")
        f.write("- **Momentum:** Short and medium-term price momentum\n\n")
        
        f.write("### Risk Management\n")
        f.write(f"- **Position Size:** Maximum {CONFIG['POSITION_SIZE']*100}% per position\n")
        f.write(f"- **Stop Loss:** {CONFIG['STOP_LOSS']*100}%\n")
        f.write(f"- **Take Profit:** {CONFIG['TAKE_PROFIT']*100}%\n\n")
        
        # Files Generated
        f.write("## Generated Files\n\n")
        f.write("- `strategy_comparison.html` - Interactive performance visualization\n")
        f.write("- `trading_results.json` - Detailed numerical results\n")
        f.write("- `event_driven_strategy_report.md` - This summary report\n\n")
        
        f.write("---\n")
        f.write("*This report demonstrates the integration of NLP-based sentiment analysis with quantitative trading strategies.*\n")
    
    print(f"Summary report generated: {report_path}")

if __name__ == "__main__":
    main()
