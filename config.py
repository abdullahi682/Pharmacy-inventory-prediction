# Configuration for Pharmacy Inventory Prediction System
thresholds = 0.32  # Kept for backward compatibility, but not used in pharmacy logic

# Pharmacy-specific configuration
EXPIRY_ALERT_DAYS = 180  # Alert 6 months before expiry
STOCKOUT_WARNING_DAYS = 30  # Warning when stock < 30 days
CRITICAL_STOCK_DAYS = 7  # Critical when stock < 7 days
DEFAULT_LEAD_TIME_DAYS = 7  # Default supplier lead time
FORECAST_HORIZONS = [7, 30, 90]  # Days for forecasting (1 week, 1 month, 3 months)


