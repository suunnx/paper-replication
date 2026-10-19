"""Paper equations (1), (2), (5), (7)–(10), with explicit timing semantics."""
import math
import numpy as np
from scipy.stats import norm, gamma


def label(demand, arrival_stock, h=1., b=9.):
    """Theorem 1: demand on [arrival, next arrival); upper endpoint at ties."""
    d = np.asarray(demand, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Demand must be a nonempty finite nonnegative vector')
    if not all(math.isfinite(x) for x in (arrival_stock, h, b)) or min(h,b)<=0:
        raise ValueError('Finite inventory and positive finite costs required')
    k = min(len(d)-1, math.floor(b*len(d)/(h+b)))
    return max(float(np.cumsum(d)[k])-arrival_stock, 0.)


def normal_order(mean, std, review, lead_mean, lead_std, inventory, h, b):
    """Equation (7), converted from base-stock level to nonnegative order."""
    protection = review + lead_mean
    if protection <= 0 or min(mean,std,lead_std) < 0:
        raise ValueError('Invalid base-stock parameters')
    level = mean*protection + norm.ppf(b/(b+h))*math.sqrt(protection*std**2 + mean**2*lead_std**2)
    return max(float(level)-inventory, 0.)


def gamma_order(mean, std, review, lead_mean, inventory, h, b):
    """Equation (8), Gamma moment fit to daily demand; deterministic limit."""
    protection = review + lead_mean
    if mean < 0 or std < 0 or protection <= 0:
        raise ValueError('Invalid Gamma parameters')
    level = mean*protection if mean == 0 or std == 0 else gamma.ppf(
        b/(b+h), protection*mean**2/std**2, scale=std**2/mean)
    return max(float(level)-inventory,0.)


def bm1_order(total_demand_quantile, inventory):
    """Equation (9): use a separately learned TOTAL demand quantile.

    Never approximate the quantile of a sum by summing daily quantiles.
    """
    return max(float(total_demand_quantile)-inventory,0.)


def bm2_order(daily_forecast, lead, review, inventory, h, b):
    """Equation (10) for an integer, fixed lead time in the executable fixture."""
    forecast=np.asarray(daily_forecast,dtype=float)
    if len(forecast)<lead+review:
        raise ValueError('Forecast does not cover protection period')
    arrival_stock=inventory-float(forecast[:lead].sum())
    return label(forecast[lead:lead+review],arrival_stock,h,b)


def step(inventory, arrivals, demand, h, b):
    """Arrivals first, then demand; unmet demand is backordered (Eqs. 1–2)."""
    end=inventory+arrivals-demand
    return end, h*max(end,0), b*max(-end,0)
