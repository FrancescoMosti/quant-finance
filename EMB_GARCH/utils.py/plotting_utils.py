import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def basic_plots(df_emb, colonna, colore, dimensioni, griglia=True):
    plt.figure(figsize=dimensioni)
    plt.plot(df_emb.index, colonna, label=colonna.name, color=colore)
    plt.xlabel("Date")
    plt.ylabel(f"{colonna.name} (USD)")
    plt.title("EMB ETF")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(griglia)
    plt.show()


def acf_plotting(data, power=1, pacf=False):
    if not pacf:
        plot_acf(data.dropna()**power, lags=30)
        plt.title("Autocorrelation Function of Squared Log Returns" if power == 2 else "Autocorrelation Function of Log Returns")
    else:
        plot_pacf(data.dropna()**power, lags=30)
        plt.title("Partial Autocorrelation Function of Squared Log Returns" if power == 2 else "Partial Autocorrelation Function of Log Returns")
    plt.show()


def plot_rolling_result(rolling_result, model_name="GARCH-t(1,2)", fig_size=(8, 4), roll=10):
    plt.figure(figsize=fig_size)
    plt.plot(rolling_result.index, rolling_result["forecast_vol"], label="Forecast Volatility", color="blue")
    plt.plot(rolling_result.index, rolling_result["realised_vol"], label="Realised Volatility", color="orange")
    plt.fill_between(rolling_result.index, rolling_result["forecast_vol"], rolling_result["realised_vol"], color='grey', alpha=0.1)
    plt.title(f"{model_name} - Rolling Forecast vs Realised Volatility ({roll} days rolling)")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def vola_plot(data, upper=None, lower=None, hist=None, model="GARCH-t(1,2)", label="forecast", start_date="2024-01-02", roll=30, frequency="D", fig_size=(8, 6)):
    forecast_start_date = pd.Timestamp(start_date)
    correct_dates = pd.date_range(start=forecast_start_date, periods=roll, freq=frequency)
    data.index = correct_dates
    if upper is not None and lower is not None:
        upper.index = correct_dates
        lower.index = correct_dates
    x_dates = mdates.date2num(data.index)

    plt.figure(figsize=fig_size)
    plt.plot(x_dates, data, label=model + " forecast", color="blue")
    if upper is not None and lower is not None:
        plt.fill_between(x_dates, lower, upper, color='blue', alpha=0.2, label="95% Confidence Interval")
    if label == "compare" and hist is not None:
        hist = hist.reindex(correct_dates).dropna()
        plt.plot(mdates.date2num(hist.index), hist, label="Realised Volatility", color="orange")
    plt.title(model + " Forecast on conditional volatility" + (" VS realised volatility" if label == "compare" else ""))
    plt.xlabel("Date")
    plt.ylabel("Volatility (Standard Deviation)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hidden_states(df_emb_hmm,hmm_model):
  n_states = df_emb_hmm["HMM_state"].nunique()
  colors = [cm.get_cmap('Dark2')(i) for i in range(n_states)] 
  plt.figure(figsize=(12, 4))
  for i in range(hmm_model.n_components):
      mask = df_emb_hmm["HMM_state"] == i
      plt.plot(df_emb_hmm.index[mask], df_emb_hmm["log_returns"][mask],
              '.', label=f"State {i}", alpha=0.6, markersize=4, color=colors[i])
  plt.title("Log-Returns identified by Hidden Markov States")
  plt.xlabel("Date")
  plt.ylabel("Log Returns")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
