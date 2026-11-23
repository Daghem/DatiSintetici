# dati-sintetici

Libreria Python per generare **serie temporali sintetiche** da usare come
dati di addestramento per modelli di machine learning.

Un dataset sintetico utile non è solo "numeri casuali": deve avere una
struttura temporale credibile (trend, stagionalità, autocorrelazione,
anomalie) perché il modello impari a generalizzare su dati che somigliano a
quelli reali. `dati-sintetici` costruisce quella struttura in modo
**dichiarativo** e **riproducibile**.

- ✅ Trend (lineare, polinomiale, esponenziale, logaritmico, sigmoidale)
- ✅ Stagionalità multi-periodo (sinusoidale, Fourier, onda quadra), additiva o moltiplicativa
- ✅ Rumore gaussiano, uniforme, t-Student, random walk, **AR(1)** autocorrelato
- ✅ Anomalie: spike, level shift, deriva
- ✅ Serie multivariate correlate tramite un fattore latente condiviso
- ✅ Feature engineering pronta per ML: lag, rolling statistics, encodings ciclici del calendario
- ✅ Split cronologico train/test (niente shuffle: si addestra sul passato e si testa sul futuro)
- ✅ Riproducibilità totale via `seed`

Dipendenze: **numpy** e **pandas**. Nient'altro.

---

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # con pytest e scikit-learn per gli esempi
```

---

## Avvio rapido

```python
from dati_sintetici import generate_time_series, ChannelSpec, TrendSpec, NoiseSpec

df = generate_time_series(
    n=365,
    freq="D",
    start="2023-01-01",
    seed=42,
    channels=[
        ChannelSpec(
            name="vendite",
            trend=TrendSpec.linear(slope=0.05, intercept=100),
            noise=NoiseSpec.ar1(scale=4.0, rho=0.9),
        )
    ],
)

df.head()
```

```
                  vendite
timestamp
2023-01-01    113.594768
2023-01-02    109.261076
2023-01-03    105.046119
2023-01-04    105.350897
2023-01-05    108.193527
```

---

## Come funziona: l'anatomia di una serie

Ogni canale è la somma di componenti indipendenti, ognuna descritta da una
dataclass:

```
valore(t) = offset + scale × ( trend(t) + stagionalità(t) + rumore(t) + anomalie(t) + fattore(t) )
```

| Componente | Classe | Cosa modella |
|---|---|---|
| Trend | `TrendSpec` | evoluzione di lungo periodo |
| Stagionalità | `SeasonalitySpec` | cicli periodici (giorno, settimana, anno) |
| Rumore | `NoiseSpec` | componente casuale, eventualmente autocorrelata |
| Anomalie | `AnomalySpec` | eventi rari e atipici |
| Canale | `ChannelSpec` | un insieme di componenti = una colonna |
| Dataset | `SeriesSpec` | indice temporale + canali + seed |

La correlazione fra i canali nasce da un **fattore latente AR(1)** condiviso:
ogni canale ci carica con un peso (`factor_loading`). Canali con lo stesso
carico risultano correlati; canali con carichi diversi no.

---

## Riferimento API

### Generazione

```python
generate_series(spec: SeriesSpec) -> pd.DataFrame
generate_time_series(n, freq, start, seed, channels, ...) -> pd.DataFrame
summarize(df) -> pd.DataFrame          # statistiche descrittive per canale
```

`generate_time_series` è un wrapper comodo; `generate_series` accetta la
dataclass `SeriesSpec` completa.

### Configurazione dei componenti

```python
from dati_sintetici import TrendSpec, SeasonalitySpec, NoiseSpec, AnomalySpec

TrendSpec.linear(slope=2.0, intercept=10.0)
TrendSpec.poly(slope=1.0, degree=3)
TrendSpec.exponential(slope=0.4, base=1.0)
TrendSpec.logistic(amplitude=10.0, center=0.5, steepness=20.0)

SeasonalitySpec.seasonal([7, 365.25], [12.0, 25.0], kind="sinus")  # kind: sinus|fourier|square
SeasonalitySpec.seasonal([24], [5.0]).multiplicative = True

NoiseSpec.gaussian(scale=2.0, loc=0.0)
NoiseSpec.ar1(scale=1.0, rho=0.85)      # rumore autocorrelato
NoiseSpec(kind="random_walk", scale=1.0)
NoiseSpec(kind="student_t", scale=1.0, df=3.0)

AnomalySpec(kind="spikes", rate=0.01, magnitude=5.0)
AnomalySpec(kind="shift", magnitude=10.0)   # salto di livello
AnomalySpec(kind="drift", magnitude=5.0)    # deriva progressiva
```

I parametri sono validati: kind sconosciuti, `rho` fuori intervallo, `degree < 1`,
canali con nomi duplicati o senza canali generano un `ValueError` esplicito.

### Periodi stagionali per frequenza

I periodi sono espressi in **numero di osservazioni**. `seasonal_periods(freq)`
restituisce i candidati corretti per la frequenza scelta:

```python
from dati_sintetici import seasonal_periods

seasonal_periods("h")     # {'daily': 24.0, 'weekly': 168.0, 'yearly': 8760.0}
seasonal_periods("D")     # {'weekly': 7.0, 'yearly': 365.25}
seasonal_periods("15min") # {'hourly': 4.0, 'daily': 96.0, 'weekly': 672.0, ...}
seasonal_periods("ME")    # {'yearly': 12.0}
```

### Preset pronti all'uso

```python
from dati_sintetici import energy_demand, web_traffic, iot_sensor, stock_price

df = generate_time_series(
    n=24 * 30, freq="h",
    channels=[energy_demand(freq="h"), web_traffic(freq="h")],
)
```

I preset sono normali `ChannelSpec`: modificali liberamente.

---

## Uso per machine learning

Il punto in cui i dati sintetici pagano: si generano infinite variazioni
"dello stesso problema" per allenare e validare un modello.

### 1. Dataset supervisionato

```python
from dati_sintetici import make_supervised

X, y = make_supervised(
    df,
    target="vendite",
    lags=(1, 2, 3, 7, 14),      # storico del target
    rolling_window=7,           # media e deviazione mobile
    rolling_stats=("mean", "std"),
    horizon=7,                  # prevedere il valore tra 7 giorni
)
```

Convenzione dei nomi delle feature:

| Nome | Significato |
|---|---|
| `vendite_lag0` | valore osservato a *t* (noto al momento della previsione) |
| `vendite_lag1` | valore a *t-1* |
| `vendite_rollmean7` | media mobile degli ultimi 7 valori |
| `vendite_rollstd7` | deviazione standard mobile |

> **Nessuna fuga di informazione**: il target di ogni riga è il valore a
> `t + horizon`, mentre tutte le feature sono calcolate su dati fino a `t`.
> Il test `test_make_supervised_no_leakage` verifica la proprietà ricostruendo
> le feature su un frame troncato.

Le colonne ausiliarie contribuiscono solo con il loro valore corrente
(`lag0`); con `lag_sources=True` ottengono anche lag e rolling.

### 2. Feature calendariali

```python
from dati_sintetici import add_time_features

df = add_time_features(df, freq="D")   # dow_sin/cos, doy_sin/cos (+ ora se intraday)
```

Sono encodings ciclici (`sin`/`cos`): il modello può imparare la stagionalità
annuale, che dai soli lag settimanali non è ricostruibile.

### 3. Classificazione

```python
from dati_sintetici import make_classification

X, y = make_classification(
    df, target="vendite", quantile=0.9, horizon=7,
    label_fn=None,      # oppure una funzione custom: (serie) -> etichette
)
```

### 4. Split cronologico

```python
from dati_sintetici import time_train_test_split

X_train, X_test = time_train_test_split(X, test_size=0.2)
```

Lo shuffle è **errato** sulle serie temporali: il test set deve essere
strictamente nel futuro rispetto al train set.

### 5. Pipeline completa

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(random_state=0)
model.fit(X_train, y.loc[X_train.index])
mae = mean_absolute_error(y.loc[X_test.index], model.predict(X_test))
```

Gli esempi completi e runnabili sono in [`examples/`](examples/):

| File | Cosa mostra |
|---|---|
| `01_generazione.py` | generazione multivariata, correlazioni, statistiche |
| `02_ml_regressione.py` | regressione a 7 giorni, confronto con la baseline "ultimo valore" |
| `03_classificazione.py` | classificazione binaria di eventi rari (F1, confusion matrix) |

Risultato tipico dell'esempio 2 (Random Forest, 1200 giorni, orizzonte 7):

```
MAE: 5.80   R2: 0.884
MAE baseline 'ultimo valore': 6.77
```

---

## Struttura del progetto

```
DatiSintetici/
├── src/dati_sintetici/
│   ├── config.py      # dataclass di configurazione (TrendSpec, ChannelSpec, ...)
│   ├── components.py  # trend, stagionalità, rumore, anomalie, fattore latente
│   ├── generator.py   # orchestrazione: SeriesSpec -> DataFrame
│   ├── features.py    # lag, rolling, calendario, dataset supervisionati
│   ├── presets.py     # configurazioni pronte (demand, traffic, sensor, price)
│   └── utils.py       # seed, indice temporale, periodi stagionali
├── tests/             # 59 test (pytest)
├── examples/          # script runnabili
├── pyproject.toml
└── README.md
```

---

## Sviluppo

```bash
pip install -e ".[dev]"
pytest                    # test
python examples/01_generazione.py
```

---

## Idee per le prossime versioni

- [ ] Serie con **cambio di regime** (markov switching) e changepoint detection
- [ ] Dipendenze fra colonne non lineari (copule gaussiane)
- [ ] Generatori di **tabelle relazionali** con chiavi esterne coerenti
- [ ] **Metriche di privacy**: k-anonimato, differenza privacy, synthetic data risk
- [ ] Adapter per SDV / Synthetic Data Vault
- [ ] Plotting helpers (matplotlib opzionale)
- [ ] CLI (`dati-sintetici genera ...`)

---

## Licenza

MIT — vedi [LICENSE](LICENSE).
