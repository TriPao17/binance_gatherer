import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm

from tools import BinanceCustomClient

# Parameters
symbol = "ETHUSDT"
max_components = 10
date_start = "2022-06-01"
date_end = "2022-09-01"
interval = "1-hour"
iterations = 1000
covariance_type = "diag"
information_criterion = "aic"


# Function to calculate aic
def calculate_aic(ll, num_params):
    aic = 2 * num_params - 2 * np.log(ll)
    return aic


# Function to calculate aic
def calculate_bic(ll, num_params, sample_size):
    aic = - 2 * np.log(ll) + num_params * np.log(sample_size)
    return aic


# Instantiate Simple API
api = BinanceCustomClient()

# Get Data
data = api.get_symbols(symbols=[symbol],
                       start_str=date_start,
                       end_str=date_end,
                       interval=interval,
                       verbose=False)

# Create Returns Column
data["return"] = data[(symbol, "close")].pct_change()
data["vol_change"] = data[(symbol, "volume")].pct_change()

# Create Nice Names
close_name = f"{symbol}_close"
volume_name = f"{symbol}_volume"

# Rename columns
data.rename(columns={(symbol, "close"): close_name,
                     (symbol, "volume"): volume_name}, inplace=True)

# Keep Relevant Columns
data = data[["close_time", close_name, volume_name, "return", "vol_change"]]

# Drop first observation
data = data.dropna()

# Loop over components
scores = []
models = []
components = []
aics = []
bics = []
for i in range(2, max_components + 1):
    # Create HMM model
    _model = hmm.GaussianHMM(n_components=i,
                             covariance_type=covariance_type,
                             n_iter=iterations)
    # Select Data
    _X = data[["return", "vol_change"]]

    # Fit model
    _model.fit(_X)

    # Get Score
    _score = _model.score(_X)

    # Calculate AIC
    _aic = calculate_aic(_score, i)

    # Calculate BIC
    _bic = calculate_bic(_score, i, data.shape[0])

    # Append Score
    scores.append(_score)

    # Append Model
    models.append(_model)

    # Append Components
    components.append(i)

    # Append Aic
    aics.append(_aic)

    # Append Aic
    bics.append(_bic)

# Get Best Model
if information_criterion == "aic":
    best_idx = aics.index(min(aics))
elif information_criterion == "bic":
    best_idx = bics.index(min(bics))
else:
    raise ValueError("Invalid Information criterion.")
best_model = models[best_idx]
best_components = components[best_idx]

# Create DataFrame
models_frame = pd.DataFrame({"components": components, "score": scores, "aic": aics, "bic": bics})

# Predict
data["state"] = best_model.predict(data[["return"]])

# Plot
for i in range(best_components):
    _data = data[data["state"] == i]
    plt.scatter(_data["close_time"], _data[close_name], label=f"State {i}", s=2, alpha=0.7)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print Transition Matrix
print(best_model.transmat_)
