"""
Pharmacy Inventory Prediction Model
This module contains the machine learning pipeline for demand forecasting
in pharmacy inventory management systems.
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from function.transformers import FeatureEngineering, WoEEncoding, ColumnSelector

# Feature columns for pharmacy inventory demand forecasting
selected_columns = [
    'current_stock', 'daily_dispensing_avg', 'monthly_dispensing_avg',
    'supplier_lead_time_days', 'DispensingRatio', 'StockDays',
    'daily_dispensing_avg_woe', 'current_stock_woe', 'StockDays_woe'
]

# Pipeline setup for pharmacy inventory demand forecasting using GradientBoostingRegressor
InventoryModel = Pipeline([
    ('feature_engineering', FeatureEngineering()),
    ('woe_encoding', WoEEncoding()),
    ('column_selector', ColumnSelector(selected_columns)),
    ('model', GradientBoostingRegressor(
          max_depth=6,
          n_estimators=300,
          random_state=42))
])
