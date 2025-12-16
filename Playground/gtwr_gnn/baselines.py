
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso
import torch
from .wls import solve_local_wls

def baseline_sklearn(name, X_train, y_train, X_all, **kwargs):
    name = name.lower()
    if name == 'ols':
        m = LinearRegression().fit(X_train, y_train)
        return m.predict(X_all)
    if name == 'huber':
        m = HuberRegressor(epsilon=kwargs.get('epsilon',1.35), alpha=kwargs.get('alpha',0.0)).fit(X_train, y_train)
        return m.predict(X_all)
    if name == 'ridge':
        m = Ridge(alpha=kwargs.get('alpha',2.0)).fit(X_train, y_train)
        return m.predict(X_all)
    if name == 'lasso':
        m = Lasso(alpha=kwargs.get('alpha',0.01)).fit(X_train, y_train)
        return m.predict(X_all)
    raise ValueError(f"Unknown baseline: {name}")

def gtwr_prior_baseline(X_t, y_t, A_prior, ridge_lambda=5.0):
    y_hat = solve_local_wls(X_t, y_t, A_prior, kind='ridge', ridge=ridge_lambda, return_betas=False)
    return y_hat
