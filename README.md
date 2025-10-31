#Kexin-Quant-Trading-Advanced  
#Multi Asset Portfolio Optimization (Pro)  
The module implements a dual-objective framework that simultaneously maximizes the Sharpe ratio and minimizes Conditional Value-at-Risk (CVaR). Using cvxpy, it visualizes the efficient frontier under various risk constraints, performs rolling backtests, and tracks portfolio weight dynamics across different levels of risk aversion.

#Volatility Modeling & Risk Forecasting  
·This module fits a GARCH(1,1) model to each asset’s return series and produces one-step-ahead volatility forecasts.
·It supports both single-fit and rolling refit modes, allowing users to balance computational speed and forecast precision.
·The script computes Value-at-Risk (VaR) and Expected Shortfall (ES) under normal or Student-t assumptions, visualizes predicted versus realized volatility, and annotates risk thresholds directly on the plots.
·Model performance metrics (AIC/BIC) and forecast statistics are automatically documented in Markdown reports, making this module a self-contained research and reporting tool for volatility forecasting and risk evaluation.

#Machine Learning Signal Strategy Backtest  
The module transforms predictive signals from machine learning models (e.g., Logistic Regression and Random Forest) into actionable trading decisions. It evaluates the resulting strategies through cumulative return analysis, annualized performance metrics, maximum drawdown, and Sharpe ratio comparison across different models.
