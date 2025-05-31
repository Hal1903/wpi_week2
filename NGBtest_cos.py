from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Generate cosinor curve data
np.random.seed(42)
n_samples = 500
X = np.linspace(0, 4 * np.pi, n_samples).reshape(-1, 1)
Y_true = np.cos(X).ravel()
noise = np.random.normal(0, 0.2, size=n_samples)
Y = Y_true + noise

# Convert to DataFrame for compatibility
X_df = pd.DataFrame(X, columns=["x"])
Y_df = pd.Series(Y, name="y")

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.3, random_state=42)

# Fit NGBoost model
ngb = NGBRegressor(
    tol = 1e-3,
    n_estimators = 500,
    validation_fraction= 0.1,
    learning_rate = 0.01,
    ).fit(X_train, Y_train)

# Predict
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# Evaluate
test_MSE = mean_squared_error(Y_preds, Y_test)
test_NLL = -Y_dists.logpdf(Y_test).mean()

print('Test MSE:', test_MSE)
print('Test NLL:', test_NLL)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_preds, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('True coordinate')
plt.ylabel('Predicted coordinate')
plt.title('NGBoost Predictions and Cosine Curve')
plt.show()


# Extract stddev safely
y_std = Y_dists.stddev
if y_std is None:
    y_std = Y_dists.scale  # fallback for Normal distribution

# Sort test data for clean line plotting
X_test_sorted = X_test.copy()
X_test_sorted['y_true'] = Y_test
X_test_sorted['y_pred'] = Y_preds
X_test_sorted['std'] = y_std 
X_test_sorted = X_test_sorted.sort_values(by='x').reset_index(drop=True)

# Extract values
x_vals = X_test_sorted['x'].values
y_true = X_test_sorted['y_true'].values
y_pred = X_test_sorted['y_pred'].values
y_std = X_test_sorted['std'].values

# Plot
plt.figure(figsize=(10, 6))

plt.scatter(x_vals, y_true, color='gray', alpha=0.4, label='True (Noisy Test Data)', zorder=1)
plt.plot(x_vals, y_pred, color='blue', label='Predicted Mean', zorder=2)
plt.fill_between(
    x_vals,
    y_pred - 2 * y_std,
    y_pred + 2 * y_std,
    color='blue',
    alpha=0.2,
    label='±2 StdDev (Confidence)',
    zorder=0
)
for i in range(len(x_vals)):
    plt.plot([x_vals[i], x_vals[i]], [y_true[i], y_pred[i]], color='red', alpha=0.3, linewidth=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('NGBoost Prediction with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.show()
