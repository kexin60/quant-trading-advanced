#Kexin-Quant-Trading-Advanced  
#Multi Asset Portfolio Optimization (Pro)  
The module implements a dual-objective framework that simultaneously maximizes the Sharpe ratio and minimizes Conditional Value-at-Risk (CVaR). Using cvxpy, it visualizes the efficient frontier under various risk constraints, performs rolling backtests, and tracks portfolio weight dynamics across different levels of risk aversion.

#Volatility Modeling & Risk Forecasting  
The module employs the GARCH(1,1) model to predict future volatility, compute Value-at-Risk (VaR) and Expected Shortfall (ES), and evaluate model performance through hit-ratio backtesting between realized and forecasted volatility.

#Machine Learning Signal Strategy Backtest  
The module transforms predictive signals from machine learning models (e.g., Logistic Regression and Random Forest) into actionable trading decisions. It evaluates the resulting strategies through cumulative return analysis, annualized performance metrics, maximum drawdown, and Sharpe ratio comparison across different models.
