import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from darts import TimeSeries
from darts.utils.statistics import plot_acf

# Hilfsfunktion für sicheren ACF-Plot
def safe_plot_acf(series, title):
    ts = TimeSeries.from_series(series)
    if len(ts) > 2:
        plot_acf(ts, max_lag=min(40, len(ts) - 1))
        plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Zu wenig Daten für ACF-Plot: {title} (n={len(ts)})")

# Daten laden
df = pd.read_csv("./data/processed_combined_df.csv", index_col="Unnamed: 0", parse_dates=True)

# Zielspalte extrahieren und Zeitzone entfernen
df_price_CH = df['CH_price [Euro]'].copy()
df_price_CH.index = df_price_CH.index.tz_localize(None)
df_price_CH.index.name = "timestamp"

# ACF auf Originaldaten (15min Auflösung)
print("===== ACF auf Originaldaten (15min Auflösung) =====")
safe_plot_acf(df_price_CH.dropna(), "ACF Strompreis CH (15min Auflösung)")

# Resampling-Konfiguration: Frequenz → Periode
resample_periods = {
    '1H': 24,     # Tagesperiodizität bei stündlichen Daten
    '1D': 7,      # Wochenperiodizität bei täglichen Daten
    '1M': 12      # Jahresperiodizität bei monatlichen Daten
}

results = {}

for freq, period in resample_periods.items():
    print(f"\n===== Analyse für Frequenz {freq}, gesamter Zeitraum =====")

    # Resample & Zeitzone entfernen
    resampled = df_price_CH.resample(freq).mean()
    resampled.index = resampled.index.tz_localize(None)

    if len(resampled.dropna()) >= period * 2:
        # Dekomposition auf kompletter Zeitreihe
        decomp = seasonal_decompose(resampled, model='additive', period=period)

        trend = decomp.trend.dropna()
        seasonal = decomp.seasonal.dropna()
        resid = decomp.resid.dropna()

        # Speichern
        results[freq] = {
            'data': resampled,
            'decomp': decomp,
            'trend': trend,
            'seasonal': seasonal,
            'resid': resid
        }

        # Komponentenplot
        fig = decomp.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"Seasonal Decomposition ({freq}, gesamter Zeitraum)", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # ACF auf kompletter Zeitreihe (resampled)
        safe_plot_acf(resampled.dropna(), f'ACF Strompreis CH (vollständig, {freq})')

        # ACF für Komponenten
        safe_plot_acf(trend, f'ACF Trend ({freq})')
        safe_plot_acf(seasonal, f'ACF Saisonalität ({freq})')
        safe_plot_acf(resid, f'ACF Residuen ({freq})')

    else:
        print(f"Nicht genug Daten für Decomposition bei {freq}: benötigt {period*2}, vorhanden {len(resampled.dropna())}")
