#Kexin-Quant-Trading-Advanced  
-[Multi_Asset_Portfolio_Optimization.py](Multi_Asset_Portfolio_Optimization.py)    
This module implements a multi-asset optimization framework that integrates mean-variance analysis, Sharpe-ratio maximization, and CVaR minimization. It generates efficient frontiers, visualizes portfolio weight dynamics under different risk levels, and supports rolling backtests with customizable constraints. All results including plots, CSV outputs, and Markdown reports—are automatically generated for transparent and reproducible portfolio analysis.    

-[Volatility_Modeling_Risk_Forecasting.py](Volatility_Modeling_Risk_Forecasting.py)     
This module fits a GARCH(1,1) model to each asset’s return series and produces one-step-ahead volatility forecasts. It supports both single-fit and rolling-refit modes to balance computational speed and precision. The script computes Value-at-Risk (VaR) and Expected Shortfall (ES) under normal or Student-t assumptions, visualizes predicted versus realized volatility, and documents model performance metrics (AIC/BIC) in Markdown reports.   

-[Machine_Learning_Signal_Strategy_Backtest.py](Machine_Learning_Signal_Strategy_Backtest.py)    
This module trains Logistic Regression and Random Forest models to predict next-day price direction using technical and factor features. Predicted signals are converted into trading positions and evaluated against a buy and hold benchmark through backtesting. The script generates cumulative return plots, detailed performance metrics, and Markdown summary reports for transparent model comparison.  

-[Quant_Trading_Engine.py](Quant_Trading_Engine.py) 
It is a modular quantitative trading framework that integrates volatility forecasting, portfolio optimization, and automated execution through the Interactive Brokers API.
