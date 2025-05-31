from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load Boston housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
Y = raw_df.values[1::2, 2]

# Convert to DataFrame to track indices
X_df = pd.DataFrame(X)
Y_df = pd.Series(Y, name="TruePrice")

# Keep original indices for tracking
X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.3, random_state=42)

# Fit NGBoost model
ngb = NGBRegressor().fit(X_train, Y_train)

# Predict
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# Evaluate
test_MSE = mean_squared_error(Y_preds, Y_test)
test_NLL = -Y_dists.logpdf(Y_test).mean()

print('Test MSE:', test_MSE)
print('Test NLL:', test_NLL)

# Create comparison DataFrame
results_df = pd.DataFrame({
    'Index': Y_test.index,
    'TruePrice': Y_test.values,
    'PredictedPrice': Y_preds,
    'LogProb': Y_dists.logpdf(Y_test),
    'StdDev': Y_dists.stddev
})

# Reset index to view row-wise
results_df = results_df.sort_values(by='Index').reset_index(drop=True)

# Display
print(results_df.head(10))  # Show first 10 comparisons

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_preds, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title('NGBoost Predictions vs True Prices')
plt.show()


# Extract stddev safely
y_std = Y_dists.stddev
if y_std is None:
    y_std = Y_dists.scale  # fallback for Normal distribution

# Add column names for plotting
X_df.columns = [f'feature_{i}' for i in range(X_df.shape[1])]
X_train.columns = X_df.columns
X_test.columns = X_df.columns

# Add prediction info
X_test_sorted = X_test.copy()
X_test_sorted['y_true'] = Y_test
X_test_sorted['y_pred'] = Y_preds
X_test_sorted['std'] = y_std

# Sort for smooth plotting
X_test_sorted = X_test_sorted.sort_values(by='y_pred').reset_index(drop=True)

# Use a feature (e.g., average rooms or just index)
x_vals = np.arange(len(X_test_sorted))  # or: X_test_sorted['feature_5'].values
y_true = X_test_sorted['y_true'].values
y_pred = X_test_sorted['y_pred'].values
y_std = X_test_sorted['std'].values

# Plot
plt.figure(figsize=(10, 6))

plt.scatter(x_vals, y_true, color='gray', alpha=0.4, label='True (Noisy Test Data)', zorder=1)
plt.plot(x_vals, y_pred, color='blue', label='Predicted Mean', zorder=2)
plt.fill_between(
    x_vals,
    y_pred - 2* y_std,
    y_pred + 2* y_std,
    color='blue',
    alpha=0.2,
    label='StdDev Envelope',
    zorder=0
)
# individual error lines between prediction line and observations
for i in range(len(x_vals)):
    plt.plot([x_vals[i], x_vals[i]], [y_true[i], y_pred[i]], color='red', alpha=0.3, linewidth=0.5)
# x unchanged -> vertical line

plt.xlabel('x')
plt.ylabel('y')
plt.title('NGBoost Prediction with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.show()


# """ approximations are deterministic unless seeds changed;
# Index  TruePrice  PredictedPrice   LogProb StdDev
# 0      0       24.0       28.273478 -7.671659   None
# 1      2       34.7       34.089068 -0.939908   None
# 2      9       18.9       18.680728 -1.469384   None
# 3     11       18.9       20.633963 -1.970303   None
# 4     18       20.2       19.305386 -1.430604   None
# 5     22       15.2       16.827916 -1.962902   None
# 6     30       12.7       13.166209 -1.255144   None
# 7     33       13.1       15.949220 -3.330912   None
# 8     39       30.8       28.600803 -4.484393   None
# 9     46       20.0       21.366448 -1.737064   None
# """