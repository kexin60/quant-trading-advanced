# Kexin-Quant-Trading-Advanced
## Overview
Advanced quantitative trading framework integrating portfolio optimization, volatility modeling, machine learning, factor models, and automated execution. This project focuses on building an end-to-end trading system with reproducible analysis, risk management, and strategy evaluation.

## Portfolio Optimization
### Multi_Asset_Portfolio_Optimization.py
-[Multi_Asset_Portfolio_Optimization.py](Multi_Asset_Portfolio_Optimization.py)    
This module implements a multi-asset optimization framework that integrates mean-variance analysis, Sharpe-ratio maximization, and CVaR minimization. It generates efficient frontiers, visualizes portfolio weight dynamics under different risk levels, and supports rolling backtests with customizable constraints. All results including plots, CSV outputs, and Markdown reports—are automatically generated for transparent and reproducible portfolio analysis.    

## Volatility & Risk Modeling
### Volatility_Modeling_Risk_Forecasting
-[Volatility_Modeling_Risk_Forecasting.py](Volatility_Modeling_Risk_Forecasting.py)     
This module fits a GARCH(1,1) model to each asset’s return series and produces one-step-ahead volatility forecasts. It supports both single-fit and rolling-refit modes to balance computational speed and precision. The script computes Value at Risk (VaR) and Expected Shortfall (ES) under normal or Student-t assumptions, visualizes predicted versus realized volatility, and documents model performance metrics (AIC/BIC) in Markdown reports.   

## Machine Learning Strategies
### Machine_Learning_Signal_Strategy_Backtest
-[Machine_Learning_Signal_Strategy_Backtest.py](Machine_Learning_Signal_Strategy_Backtest.py)    
This module trains Logistic Regression and Random Forest models to predict next-day price direction using technical and factor features. Predicted signals are converted into trading positions and evaluated against a buy and hold benchmark through backtesting. The script generates cumulative return plots, detailed performance metrics, and Markdown summary reports for transparent model comparison.  

## Factor Models
### Factor_Based_Return_Prediction_Model
-[Quant_Trading_Engine.py](Quant_Trading_Engine.py)   
This module serves as the core engine of the framework, integrating all functional components volatility forecasting, signal generation, portfolio optimization, and trade execution. It orchestrates real-time data flow, manages model updates, and automates trading via the Interactive Brokers API. Comprehensive trade logs, performance summaries, and diagnostic plots are generated for end-to-end transparency and reproducibility.

## Options & Volatility Surface
### Implied_Volatility_Surface_Modeling
-[Factor_Based_Return_Prediction_Model.py](Factor_Based_Return_Prediction_Model.py)  
Builds a multi-factor regression model to predict next-day excess returns using Fama-French factors, momentum, volatility, beta, and sentiment features. Supports linear and tree based models (OLS, Lasso, Ridge, Random Forest, XGBoost, LightGBM). Generates performance metrics, exposure plots, and Markdown summary reports for transparent evaluation.

## Event-Driven Trading
### Event_Driven_Trading_Strategy_with_News_Sentiment
-[Implied_Volatility_Surface_Modeling.py](Implied_Volatility_Surface_Modeling.py)    
This module constructs and analyzes the implied volatility surface from option market data. It retrieves option chains via yfinance or IBKR API, computes implied volatilities using the Black-Scholes model, and fits SABR curves to capture volatility smiles. The script produces 3D surface plots, volatility term structures, and skew evolution metrics, supporting option risk modeling and visualization.  

## Trading Engine
### Quant_Trading_Engine
-[Event_Driven_Trading_Strategy_with_News_Sentiment.py](Event_Driven_Trading_Strategy_with_News_Sentiment.py)    
This module implements an event driven trading framework that fuses real-time news sentiment (FinBERT) with technical and momentum signals. It aggregates news from multiple APIs, computes sentiment scores, and dynamically generates trading decisions with integrated risk control and backtesting. The script compares sentiment aware versus technical only strategies, producing performance metrics, trade logs, and interactive visualizations.  
### Quant_Trading_Engine_Plus
-[Quant_Trading_Engine_Plus.py](Quant_Trading_Engine_Plus.py)    
This module serves as the enhanced and unified core of the entire quantitative trading framework, integrating all functional components data ingestion, signal generation, risk management, portfolio optimization, and automated execution via the Interactive Brokers (IBKR) API into one cohesive pipeline. It orchestrates real-time data flow, merges multi-source signals from machine learning, factor models, and news sentiment analysis, and dynamically generates portfolio allocations under risk and turnover constraints. The engine supports both paper trading and live execution, automatically generating daily trade reports, performance summaries, and risk analytics for transparent evaluation.  

## Environment Setup 
python3 -m venv .venv  
source .venv/bin/activate # macOS / Linux  
.venv\Scripts\activate # Windows  
pip install -r requirements.txt  
