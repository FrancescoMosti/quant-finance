import numpy as np
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro


def garch_fun(my_data, my_p, my_q, vol_model, distribuzione="normal", force_omega=False):
    garch_model = arch_model(my_data.dropna(), vol=vol_model, p=my_p, q=my_q, dist=distribuzione)
    garch_fit = garch_model.fit(disp="off")
    if force_omega:
        garch_fit.params["omega"] = 0
    return garch_fit


def residual_analysis(my_model, qq="normal"):
    std_residuals = my_model.std_resid
    sns.histplot(std_residuals, kde=True)
    plt.title("Distribution of Standardized Residuals")
    plt.show()
    acf_plotting(std_residuals, power=1)
    acf_plotting(std_residuals, power=1, pacf=True)
    if qq == "normal":
        stats.probplot(std_residuals, dist="norm", plot=plt)
    else:
        stats.probplot(std_residuals, dist=stats.t(df=my_model.params["nu"]), plot=plt)
    stat, p = shapiro(std_residuals)
    print("\nShapiro-Wilk Test Statistic: {:.4f}, P-value: {:.4f}".format(stat, p))
    print("\nLjung-Box Test:\n{}".format(acorr_ljungbox(std_residuals, lags=10)))


def forecasting_with_ci(my_model, start_date="2024-01-02", forecast_hor=30, trad_d=1):
    vol_forecast = my_model.forecast(horizon=forecast_hor, reindex=False)
    var = vol_forecast.variance.iloc[-1]
    sigma = np.sqrt(var) * np.sqrt(trad_d)
    upper = sigma * 1.05
    lower = sigma * 0.95
    dates = pd.date_range(start=pd.Timestamp(start_date), periods=forecast_hor, freq="D").normalize()
    sigma.index = dates
    upper.index = dates
    lower.index = dates
    return sigma, upper, lower


def rolling_garch_forecast(returns, my_p=1, my_q=1, my_dist="t", model_vol="GARCH", window_size=500, forecast_hor=1, step=1, trad_d=252, start_date=None, end_date=None):
    if start_date:
        returns = returns[returns.index >= pd.to_datetime(start_date)]
    if end_date:
        returns = returns[returns.index <= pd.to_datetime(end_date)]
    forecast_list = []
    realised_list = []
    date_list = []
    for i in range(window_size, len(returns) - forecast_hor, step):
        train = returns.iloc[i - window_size:i]
        test_window = returns.iloc[i:i + forecast_hor]
        model = arch_model(train, p=my_p, q=my_q, vol=model_vol, dist=my_dist)
        fit = model.fit(disp="off")
        forecast = fit.forecast(horizon=forecast_hor, reindex=False)
        sigma_f = np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(trad_d)
        if forecast_hor == 1:
            future_window = returns.iloc[i + forecast_hor: i + forecast_hor + 3]
            realised = future_window.std() * np.sqrt(trad_d) if len(future_window) >= 2 else np.nan
        else:
            realised = test_window.rolling(window=forecast_hor).std().iloc[-1] * np.sqrt(trad_d)
        forecast_list.append(sigma_f.mean())
        realised_list.append(realised)
        date_list.append(test_window.index[-1])
    return pd.DataFrame({"date": date_list, "forecast_vol": forecast_list, "realised_vol": realised_list}).set_index("date")

def index_comp(name_index):
  idx_raw = yf.download(name_index, start="2021-01-01", end="2024-01-01")
  idx = idx_raw.copy()
  idx.columns = idx.columns.get_level_values(0)
  idx = idx[["Close"]].rename(columns={"Close": name_index})
  idx.index = idx.index.tz_localize(None).normalize()
  df_emb_idx = df_emb[["log_returns"]].copy()
  df_merged = df_emb_idx.join(idx, how="inner")
  rolling_corr = df_merged["log_returns"].rolling(window=30).corr(df_merged[name_index])
  plt.figure(figsize=(10, 4))
  plt.plot(rolling_corr.index, rolling_corr, color="purple", label="30d Rolling Correlation")
  plt.axhline(0, linestyle="--", color="black", alpha=0.5)
  plt.title(f"30-Day Rolling Correlation between EMB Log Returns and {name_index}")
  plt.xlabel("Date")
  plt.ylabel("Correlation")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  static_corr = df_merged["log_returns"].corr(df_merged[name_index])
  print(f"Overall correlation: {static_corr:.4f}")
