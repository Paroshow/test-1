# ===============================
# Packages
# ===============================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import os


# ===============================
# Data loading (Bloomberg CSVs)
# ===============================

def port_data_load_prices_bbg(assets_list, start_date_text, end_date_text, data_directory="bloomberg_data"):
    """
    Load adjusted daily close prices for multiple assets from local Bloomberg CSVs.
    Each CSV has two columns without headers:
        [0] Date (format: dd.mm.yyyy)
        [1] Adjusted close price

    Returns a clean DataFrame of aligned, forward-filled prices over the given range.
    Prints diagnostics on true missing values and final coverage.
    """
    print("\n--- Bloomberg Price Data Loading ---")

    all_data = {}
    start_date = pd.to_datetime(start_date_text)
    end_date = pd.to_datetime(end_date_text)

    for asset in assets_list:
        file_name = f"{asset.lower()}.csv"
        file_path = os.path.join(data_directory, file_name)
        if not os.path.exists(file_path):
            print(f"[WARN] Missing file for {asset}: {file_path}")
            continue

        try:
            frame = pd.read_csv(file_path, header=None, names=["date", asset])
            frame["date"] = pd.to_datetime(frame["date"], format="%d.%m.%Y", errors="coerce")
            frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
            frame = frame.loc[(frame.index >= start_date) & (frame.index <= end_date)]

            # Check for true NaNs
            nan_rows = frame[asset].isna()
            if nan_rows.any():
                missing_dates = frame.index[nan_rows]
                first_dates = missing_dates[:3].strftime("%Y-%m-%d").tolist()
                last_dates = missing_dates[-3:].strftime("%Y-%m-%d").tolist()
                print(f"[WARN] {asset}: {nan_rows.sum()} missing values | first: {first_dates} | last: {last_dates}")

            all_data[asset] = frame[asset].astype(float)

        except Exception as e:
            print(f"[ERROR] Failed to load {asset}: {e}")

    if not all_data:
        raise ValueError("[ERROR] No valid CSV files loaded.")

    # Combine and clean
    data_all = pd.concat(all_data, axis=1).sort_index()
    data_all = data_all.ffill().dropna(how="all")

    # Final diagnostics
    first_date = data_all.index.min().strftime("%Y-%m-%d")
    last_date = data_all.index.max().strftime("%Y-%m-%d")
    nan_counts = data_all.isna().sum()
    tickers_with_nans = nan_counts[nan_counts > 0]

    print(f"[INFO] Date range: {first_date} → {last_date}")

    if not tickers_with_nans.empty:
        print("[WARN] Remaining NaNs detected after alignment (forward-filled):")
        for ticker, count in tickers_with_nans.items():
            missing_dates = data_all.index[data_all[ticker].isna()]
            first_dates = missing_dates[:3].strftime("%Y-%m-%d").tolist()
            last_dates = missing_dates[-3:].strftime("%Y-%m-%d").tolist()
            print(f"       {ticker}: {count} NaNs | first: {first_dates} | last: {last_dates}")
    else:
        print("[INFO] No NaNs detected after alignment. Data fully continuous.")

    print(f"[INFO] Loaded tickers: {list(data_all.columns)}")
    print("[INFO] Missing values handled using forward fill.")
    print("--- End Bloomberg Price Data Loading ---\n")

    return data_all


def port_data_compute_returns(prices_frame):
    """
    Compute daily percentage returns from Bloomberg price data.
    Since prices are already cleaned and forward-filled, this function performs no diagnostics.
    """
    returns_frame = prices_frame.pct_change().dropna(how="any")
    return returns_frame


def port_data_load_benchmark_bbg(data_directory="bloomberg_data", file_name="benchmark.csv",
                                 start_date_text=None, end_date_text=None):
    """
    Load benchmark CSV in Bloomberg 2-column format (Date + Price).
    Detects missing values, applies forward fill, and computes daily returns.
    Returns: pd.Series of benchmark returns indexed by date.
    """
    print("\n--- Bloomberg Benchmark Loading ---")
    file_path = os.path.join(data_directory, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Benchmark file not found: {file_path}")

    frame = pd.read_csv(file_path, header=None, names=["date", "price"])
    frame["date"] = pd.to_datetime(frame["date"], format="%d.%m.%Y", errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    frame["price"] = frame["price"].astype(float)

    if start_date_text:
        start_date = pd.to_datetime(start_date_text)
        frame = frame.loc[frame.index >= start_date]
    if end_date_text:
        end_date = pd.to_datetime(end_date_text)
        frame = frame.loc[frame.index <= end_date]

    # Detect NaNs before cleaning
    nan_rows = frame["price"].isna()
    if nan_rows.any():
        missing_dates = frame.index[nan_rows]
        first_dates = missing_dates[:3].strftime("%Y-%m-%d").tolist()
        last_dates = missing_dates[-3:].strftime("%Y-%m-%d").tolist()
        print(f"[WARN] Benchmark: {nan_rows.sum()} missing values | first: {first_dates} | last: {last_dates}")
    else:
        print("[INFO] No NaNs detected. Data fully continuous.")

    # Forward-fill and compute returns
    frame["price"] = frame["price"].ffill()
    returns = frame["price"].pct_change().dropna()
    returns.name = "benchmark_return"

    first_date = returns.index.min().strftime("%Y-%m-%d")
    last_date = returns.index.max().strftime("%Y-%m-%d")

    print(f"[INFO] Date range: {first_date} → {last_date}")
    print("[INFO] Missing values handled using forward fill.")
    print("--- End Bloomberg Benchmark Loading ---\n")

    return returns


# ===============================
# Asset-level summary (with RF annual, proper annual Sharpe)
# ===============================

def port_data_summary_assets(returns_frame, benchmark_return_series=None,
                             risk_free_rate_annual=0.0, trading_days_per_year=252,
                             split_date_text=None):
    """
    Summarize per-asset annualized performance metrics.
    If split_date_text is provided, only data up to that date is used (in-sample period).
    """
    if split_date_text:
        split_date = pd.to_datetime(split_date_text)
        returns_frame = returns_frame.loc[returns_frame.index <= split_date]
        if benchmark_return_series is not None:
            benchmark_return_series = benchmark_return_series.loc[benchmark_return_series.index <= split_date]
        print(f"\n[INFO] Using in-sample period up to {split_date.strftime('%Y-%m-%d')} for analysis.")
    else:
        print("\n[INFO] Using entire available period for analysis.")

    start_date = returns_frame.index.min().strftime("%Y-%m-%d")
    end_date = returns_frame.index.max().strftime("%Y-%m-%d")

    print("\n--- Asset Annualized Performance Summary ---")
    print(f"[INFO] Analysis period: {start_date} → {end_date}\n")

    mean_daily = returns_frame.mean()
    vol_daily = returns_frame.std(ddof=0)
    rf_daily = risk_free_rate_annual / trading_days_per_year

    mean_annual = mean_daily * trading_days_per_year
    vol_annual = vol_daily * np.sqrt(trading_days_per_year)

    sharpe_annual = (mean_annual - risk_free_rate_annual) / vol_annual.replace(0.0, np.nan)
    downside_std_daily = returns_frame.sub(rf_daily).clip(upper=0.0).std(ddof=0)
    downside_vol_annual = downside_std_daily * np.sqrt(trading_days_per_year)
    sortino_annual = (mean_annual - risk_free_rate_annual) / downside_vol_annual.replace(0.0, np.nan)

    summary_frame = pd.DataFrame({
        "Mean (Ann.)": mean_annual,
        "Vol (Ann.)": vol_annual,
        "Sharpe (Ann.)": sharpe_annual,
        "Sortino (Ann.)": sortino_annual,
    })

    if benchmark_return_series is not None:
        bench_mean_d = benchmark_return_series.mean()
        bench_vol_d = benchmark_return_series.std(ddof=0)
        bench_mean_a = bench_mean_d * trading_days_per_year
        bench_vol_a = bench_vol_d * np.sqrt(trading_days_per_year)

        bench_sharpe_a = (bench_mean_a - risk_free_rate_annual) / bench_vol_a
        bench_downside_d = benchmark_return_series.sub(rf_daily).clip(upper=0.0).std(ddof=0)
        bench_sortino_a = (bench_mean_a - risk_free_rate_annual) / (bench_downside_d * np.sqrt(trading_days_per_year))

        active_daily = returns_frame.sub(benchmark_return_series, axis=0)
        te_annual = active_daily.std(ddof=0) * np.sqrt(trading_days_per_year)
        info_ratio = (mean_annual - bench_mean_a) / te_annual.replace(0.0, np.nan)

        summary_frame["TE (Ann.)"] = te_annual
        summary_frame["Info (Ann.)"] = info_ratio
        summary_frame.loc["Benchmark"] = [
            bench_mean_a, bench_vol_a, bench_sharpe_a, bench_sortino_a, np.nan, np.nan
        ]

    summary_frame = summary_frame.round(6)
    print(summary_frame)
    print(f"\n[INFO] rf_annual = {risk_free_rate_annual:.6f} | rf_daily = {rf_daily:.8f}")
    return summary_frame


# ===============================
# Helper functions (metrics)
# ===============================

def port_metrics_compute_portfolio_series(weights_series, returns_frame):
    """
    Compute portfolio daily returns as weighted sum of asset returns.
    Ensures weights align with return columns.
    """
    weights_series = weights_series.reindex(returns_frame.columns).fillna(0.0)
    portfolio_return_series = (returns_frame * weights_series).sum(axis=1)
    return portfolio_return_series


def port_metrics_expected_annual_return(portfolio_return_series, trading_days_per_year):
    """
    Compute annualized expected return from average daily return.
    """
    return float(portfolio_return_series.mean() * trading_days_per_year)


def port_metrics_volatility_daily(portfolio_return_series):
    """
    Compute daily portfolio volatility (standard deviation of returns).
    """
    return float(portfolio_return_series.std(ddof=0))


def port_metrics_volatility_annual(portfolio_return_series, trading_days_per_year):
    """
    Compute annualized volatility using sqrt(time) scaling.
    """
    return float(portfolio_return_series.std(ddof=0) * np.sqrt(trading_days_per_year))


def port_metrics_sharpe_ratio(portfolio_return_series, risk_free_rate_annual, trading_days_per_year):
    """
    Compute annualized Sharpe ratio (excess return / total volatility).
    """
    rf_daily = risk_free_rate_annual / trading_days_per_year
    excess_daily = portfolio_return_series - rf_daily
    avg_excess_daily = excess_daily.mean()
    std_daily = portfolio_return_series.std(ddof=0)
    if std_daily == 0:
        return np.nan
    return float((avg_excess_daily * trading_days_per_year) / (std_daily * np.sqrt(trading_days_per_year)))


def port_metrics_sortino_ratio(portfolio_return_series, risk_free_rate_annual, trading_days_per_year):
    """
    Compute annualized Sortino ratio (excess return / downside volatility).
    Downside deviation considers only negative excess returns.
    """
    rf_daily = risk_free_rate_annual / trading_days_per_year
    downside = np.minimum(0.0, portfolio_return_series - rf_daily)
    downside_std_daily = downside.std(ddof=0)
    if downside_std_daily == 0:
        return np.nan
    avg_excess_daily = (portfolio_return_series - rf_daily).mean()
    return float((avg_excess_daily * trading_days_per_year) / (downside_std_daily * np.sqrt(trading_days_per_year)))


def port_metrics_drawdown_max(portfolio_return_series): 
    """
    Compute maximum drawdown from cumulative portfolio wealth.
    """
    wealth = (1.0 + portfolio_return_series).cumprod()
    rolling_max = wealth.cummax()
    drawdown_series = wealth / rolling_max - 1.0
    return float(drawdown_series.min())


def port_metrics_high_water_mark(portfolio_return_series):
    """
    Compute the High-Water Mark (HWM) time series.
    HWM_t = maximum cumulative portfolio value achieved up to time t.
    Returns a pandas Series aligned with the input.
    """
    wealth = (1.0 + portfolio_return_series).cumprod()
    hwm_series = wealth.cummax()
    return hwm_series


def port_metrics_tracking_error(portfolio_return_series, benchmark_return_series, trading_days_per_year):
    """
    Compute annualized Tracking Error (standard deviation of active returns).
    """
    active_daily = portfolio_return_series - benchmark_return_series
    te_annual = active_daily.std(ddof=0) * np.sqrt(trading_days_per_year)
    return float(te_annual)


def port_metrics_information_ratio(portfolio_return_series, benchmark_return_series, trading_days_per_year):
    """
    Compute annualized Information Ratio relative to benchmark.
    """
    te_annual = port_metrics_tracking_error(portfolio_return_series, benchmark_return_series, trading_days_per_year)
    if te_annual == 0:
        return np.nan
    port_ann = portfolio_return_series.mean() * trading_days_per_year
    bench_ann = benchmark_return_series.mean() * trading_days_per_year
    return float((port_ann - bench_ann) / te_annual)


def port_metrics_summarize(weights_series, returns_frame, benchmark_return_series, risk_free_rate_annual, trading_days_per_year):
    """
    Compute all main performance metrics for a portfolio.
    """
    port_series = port_metrics_compute_portfolio_series(weights_series, returns_frame)
    te_annual = port_metrics_tracking_error(port_series, benchmark_return_series, trading_days_per_year)
    info_ratio = port_metrics_information_ratio(port_series, benchmark_return_series, trading_days_per_year)

    return {
        "expected_return_annual": port_metrics_expected_annual_return(port_series, trading_days_per_year),
        "volatility_daily": port_metrics_volatility_daily(port_series),
        "volatility_annual": port_metrics_volatility_annual(port_series, trading_days_per_year),
        "sharpe_ratio": port_metrics_sharpe_ratio(port_series, risk_free_rate_annual, trading_days_per_year),
        "sortino_ratio": port_metrics_sortino_ratio(port_series, risk_free_rate_annual, trading_days_per_year),
        "max_drawdown": port_metrics_drawdown_max(port_series),
        "tracking_error_annual": te_annual,
        "information_ratio": info_ratio,
    }


def port_utils_clean_weights(weights_dict):
    """
    Convert weight dictionary to sorted pandas Series and remove near-zero allocations.
    """
    series = pd.Series(weights_dict)
    series = series[series.abs() > 1e-6]
    return series.sort_values(ascending=False)


# ===============================
# Present results (clean & organized)
# ===============================

def port_table_build_weights(weights_series, portfolio_name_text):
    """Convert weights Series to a single-column DataFrame."""
    frame = weights_series.to_frame(name=portfolio_name_text)
    frame.index.name = "asset"
    return frame


def port_table_build_metrics(metrics_dict, portfolio_name_text):
    """Convert metrics dictionary to single-row DataFrame."""
    return pd.DataFrame([metrics_dict], index=[portfolio_name_text])


