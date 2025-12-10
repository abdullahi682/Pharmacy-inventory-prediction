import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from function.transformers import FeatureEngineering, WoEEncoding, ColumnSelector
from datetime import datetime, timedelta

# Load real pharmacy inventory dataset for training
data = pd.read_csv('datasets/pharmacy_inventory_dataset.csv')

# Process daily usage history for time series modeling
time_series_data = {}
time_series_models = {}

for _, row in data.iterrows():
    product_id = row['product_id']
    # Parse the daily usage string into a list of floats
    daily_usage_str = str(row['daily_usage_last_30_days'])
    try:
        daily_usage = [float(x.strip()) for x in daily_usage_str.split(',') if x.strip()]
        if len(daily_usage) >= 7:  # Need at least a week of data
            time_series_data[product_id] = pd.Series(daily_usage, name='daily_usage')

            # Train ARIMA and Exponential Smoothing models for this product
            try:
                from statsmodels.tsa.arima.model import ARIMA
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                # ARIMA model
                arima_model = ARIMA(daily_usage, order=(1, 1, 1))
                arima_fit = arima_model.fit()

                # Exponential Smoothing model
                es_model = ExponentialSmoothing(daily_usage, seasonal_periods=7, trend='add', seasonal='add')
                es_fit = es_model.fit()

                time_series_models[product_id] = {
                    'arima': arima_fit,
                    'exponential_smoothing': es_fit,
                    'historical_data': daily_usage
                }
            except Exception as e:
                print(f"Failed to train models for {product_id}: {e}")
                time_series_models[product_id] = {
                    'arima': None,
                    'exponential_smoothing': None,
                    'historical_data': daily_usage
                }
    except Exception as e:
        print(f"Failed to process daily usage for {product_id}: {e}")
        continue

print(f"Loaded time series data for {len(time_series_data)} products")
print(f"Trained models for {len(time_series_models)} products")

# Save time series models
joblib.dump(time_series_models, 'pharmacy_time_series_models.joblib')
joblib.dump(data, 'pharmacy_products_data.joblib')

# For backward compatibility, create a simple regression model
X = data[['current_stock', 'daily_dispensing_avg', 'monthly_dispensing_avg', 'supplier_lead_time_days']]
y = data['daily_dispensing_avg']  # Use average demand as target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple model for demand forecasting
demand_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# Train the demand forecasting model
demand_model.fit(X_train, y_train)

# Evaluate the model
y_pred = demand_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Save the trained demand model
joblib.dump(demand_model, 'pharmacy_inventory_prediction_pipeline.joblib')

# Create category-based fallback logic using real data
category_averages = {}
for category in data['category'].unique():
    cat_products = data[data['category'] == category]
    if len(cat_products) > 0:
        category_averages[category.lower()] = {
            'avg_daily_demand': cat_products['daily_dispensing_avg'].mean(),
            'avg_monthly_demand': cat_products['monthly_dispensing_avg'].mean(),
            'avg_lead_time': cat_products['supplier_lead_time_days'].mean(),
            'demand_std': cat_products['daily_dispensing_avg'].std()
        }

# Save category averages for new product fallback
joblib.dump(category_averages, 'pharmacy_category_averages.joblib')

# Evaluate time series models and calculate performance metrics
time_series_performance = {}

for product_id, models in time_series_models.items():
    if models['arima'] is not None and models['exponential_smoothing'] is not None:
        historical_data = models['historical_data']

        # Split data for validation (use last 20% for testing)
        train_size = int(len(historical_data) * 0.8)
        train_data = historical_data[:train_size]
        test_data = historical_data[train_size:]

        if len(test_data) > 0:
            # ARIMA predictions
            try:
                arima_pred = models['arima'].forecast(len(test_data))
                arima_mae = mean_absolute_error(test_data, arima_pred)
                arima_rmse = mean_squared_error(test_data, arima_pred, squared=False)
                arima_mape = mean_absolute_percentage_error(test_data, arima_pred) * 100
                arima_aic = models['arima'].aic
            except:
                arima_mae = arima_rmse = arima_mape = arima_aic = float('inf')

            # Exponential Smoothing predictions
            try:
                es_pred = models['exponential_smoothing'].forecast(len(test_data))
                es_mae = mean_absolute_error(test_data, es_pred)
                es_rmse = mean_squared_error(test_data, es_pred, squared=False)
                es_mape = mean_absolute_percentage_error(test_data, es_pred) * 100
                # AIC equivalent for ES (simplified)
                es_aic = len(train_data) * np.log(np.var(models['exponential_smoothing'].resid)) + 2 * 3
            except:
                es_mae = es_rmse = es_mape = es_aic = float('inf')

            time_series_performance[product_id] = {
                'arima': {
                    'mae': arima_mae,
                    'rmse': arima_rmse,
                    'mape': arima_mape,
                    'aic': arima_aic
                },
                'exponential_smoothing': {
                    'mae': es_mae,
                    'rmse': es_rmse,
                    'mape': es_mape,
                    'aic': es_aic
                }
            }

# Save performance metrics
joblib.dump(time_series_performance, 'pharmacy_model_performance.joblib')

print(f"Trained models for {len(time_series_models)} products")
print(f"Performance metrics calculated for {len(time_series_performance)} products")

# Save last training timestamp for scheduled retraining
training_metadata = {
    'last_training_date': datetime.now(),
    'n_products': len(time_series_models),
    'model_version': '1.0',
    'training_parameters': {
        'arima_order': (1, 1, 1),
        'es_seasonal_periods': 7,
        'es_trend': 'add',
        'es_seasonal': 'add'
    }
}
joblib.dump(training_metadata, 'training_metadata.joblib')

# Update model reference in loader.py
# Note: In production, this would be handled differently

# Additional logic for inventory predictions
def forecast_demand(product_id, days_ahead=30):
    """Forecast demand for a product over the next days_ahead days"""
    # Placeholder: use historical data and model to forecast
    # In real implementation, this would use time series features
    base_demand = inventory_data[inventory_data['product_id'] == product_id]['daily_dispensing_avg'].iloc[0]
    # Simple exponential smoothing or model-based forecast
    forecast = [base_demand * (1 + 0.05 * i) for i in range(days_ahead)]  # Placeholder trend
    return forecast

def calculate_depletion_date(current_stock, daily_avg):
    """Calculate estimated depletion date"""
    if daily_avg <= 0:
        return None
    days_to_deplete = current_stock / daily_avg
    return current_date + timedelta(days=days_to_deplete)

def check_expiry_risk(batch_expiry_dates, depletion_date):
    """Check if stock will expire before depletion"""
    if depletion_date is None:
        return False
    for expiry in batch_expiry_dates:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        if expiry_date < depletion_date:
            return True
    return False

def check_stockout_risk(current_stock, daily_avg, lead_time_days):
    """Check if stockout will occur before replenishment"""
    depletion_date = calculate_depletion_date(current_stock, daily_avg)
    if depletion_date is None:
        return False
    replenishment_date = current_date + timedelta(days=int(lead_time_days))
    return depletion_date < replenishment_date

def calculate_last_order_date(lead_time_days, depletion_date):
    """Calculate the last date to order to avoid stockout"""
    if depletion_date is None:
        return None
    return depletion_date - timedelta(days=lead_time_days)

# Load inventory data and current date (assuming it's defined in loader.py)
from loader import inventory_data, current_date
