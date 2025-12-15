import pandas as pd

def load_sales_data(file_path):
    """
    Load and preprocess the e-commerce sales CSV.
    Expects columns: ['date', 'sales', ... optional covariates]
    """
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df
import pandas as pd

def create_features(df, lags=[1,7], rolling_windows=[7,14]):
    """
    Add calendar, lag, and rolling features to dataframe.
    """
    df = df.copy()
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # Lag features
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    
    # Rolling features
    for window in rolling_windows:
        df[f'sales_roll_mean_{window}'] = df['sales'].shift(1).rolling(window).mean()
        df[f'sales_roll_std_{window}'] = df['sales'].shift(1).rolling(window).std()
    
    df = df.dropna().reset_index(drop=True)
    return df
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_val_test_split(df, target_col='sales', train_frac=0.7, val_frac=0.15):
    """
    Chronological train/validation/test split to avoid leakage
    """
    n = len(df)
    train_end = int(n*train_frac)
    val_end = int(n*(train_frac+val_frac))
    
    X = df.drop(columns=[target_col, 'date'])
    y = df[target_col]
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def naive_baseline(y_train, y_val, lag=1):
    """
    Predict using the last observed sales value
    """
    y_pred = np.roll(y_val, lag)
    y_pred[:lag] = y_train[-lag:]  # fill first prediction
    return y_pred

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return mae, rmse, r2
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def train_mlp(X_train, y_train, X_val, y_val):
    """
    Train an MLPRegressor with simple hyperparameter tuning
    """
    mlp = MLPRegressor(random_state=42, early_stopping=True, max_iter=500)
    
    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (32,16)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(mlp, param_grid, cv=tscv, scoring='r2')
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    
    # Validation evaluation
    y_val_pred = grid.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    r2 = r2_score(y_val, y_val_pred)
    print(f"Validation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    return grid.best_estimator_

def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def final_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("Final Test Set Evaluation:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    plt.figure(figsize=(12,5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title("Predicted vs Actual Sales")
    plt.xlabel("Time")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()
    
    return y_pred
from src.data_load import load_sales_data
from src.features import create_features
from src.preprocess import train_val_test_split
from src.baseline import naive_baseline, evaluate_regression
from src.model_mlp import train_mlp, plot_predictions
from src.evaluate import final_evaluation

# Load data
df = load_sales_data('data/raw/sales.csv')

# Feature engineering
df_feat = create_features(df)

# Train/val/test split
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_feat)

# Baseline
print("Baseline evaluation:")
y_base_pred = naive_baseline(y_train, y_val, lag=1)
evaluate_regression(y_val, y_base_pred)

# Train MLP
mlp_model = train_mlp(X_train, y_train, X_val, y_val)

# Final evaluation
final_preds = final_evaluation(mlp_model, X_test, y_test)
