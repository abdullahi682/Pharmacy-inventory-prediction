#!/usr/bin/env python3
"""
Automated Model Retraining System for Pharmacy Inventory Prediction
Retrains models when performance degrades or new data becomes available.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ModelRetrainer:
    def __init__(self):
        self.models = None
        self.products_data = None
        self.performance_metrics = None
        self.load_current_models()

    def load_current_models(self):
        """Load current trained models and data"""
        try:
            self.models = joblib.load('pharmacy_time_series_models.joblib')
            self.products_data = joblib.load('pharmacy_products_data.joblib')
            self.performance_metrics = joblib.load('pharmacy_model_performance.joblib')
            print(f"Loaded current models for {len(self.models)} products")
        except Exception as e:
            print(f"Error loading current models: {e}")
            return False
        return True

    def check_retraining_needed(self):
        """Check if retraining is needed based on various criteria"""
        retraining_needed = []

        # Check training age
        try:
            metadata = joblib.load('training_metadata.joblib')
            last_training = metadata['last_training_date']
            days_since_training = (datetime.now() - last_training).days

            if days_since_training >= 30:
                print(f"Models are {days_since_training} days old - retraining recommended")
                return True
        except:
            print("Could not check training metadata - retraining recommended")
            return True

        # Check performance degradation
        for product_id in self.products_data['product_id'].values:
            if product_id in self.performance_metrics:
                metrics = self.performance_metrics[product_id]

                # Check if recent performance is worse than baseline
                if self._has_performance_degraded(product_id, metrics):
                    retraining_needed.append(product_id)

        if retraining_needed:
            print(f"Performance degradation detected for {len(retraining_needed)} products")
            return True

        # Check for new products
        new_products = self._find_new_products()
        if new_products:
            print(f"Found {len(new_products)} new products requiring training")
            return True

        return False

    def _has_performance_degraded(self, product_id, metrics):
        """Check if model performance has degraded"""
        # Simplified check - in production, would compare recent validation metrics
        # with baseline performance stored in metadata
        return False  # Placeholder

    def _find_new_products(self):
        """Find products that don't have trained models"""
        if not self.models:
            return list(self.products_data['product_id'].values)

        trained_products = set(self.models.keys())
        all_products = set(self.products_data['product_id'].values)

        return list(all_products - trained_products)

    def retrain_models(self):
        """Retrain all models with updated data"""
        print(f"Starting model retraining at {datetime.now()}")

        # Generate synthetic time series data for each product
        time_series_data = self._generate_time_series_data()

        # Train models for each product
        updated_models = {}
        updated_performance = {}

        for product_id in self.products_data['product_id'].values:
            try:
                if product_id in time_series_data:
                    print(f"Retraining model for product {product_id}")

                    # Train ARIMA model
                    arima_model = self._train_arima(time_series_data[product_id])

                    # Train Exponential Smoothing model
                    es_model = self._train_exponential_smoothing(time_series_data[product_id])

                    # Evaluate models
                    arima_metrics = self._evaluate_model(arima_model, time_series_data[product_id])
                    es_metrics = self._evaluate_model(es_model, time_series_data[product_id])

                    # Store best model
                    if arima_metrics['aic'] < es_metrics['aic']:
                        best_model = arima_model
                        best_metrics = arima_metrics
                        model_type = 'arima'
                    else:
                        best_model = es_model
                        best_metrics = es_metrics
                        model_type = 'exponential_smoothing'

                    updated_models[product_id] = {
                        'model': best_model,
                        'model_type': model_type,
                        'trained_date': datetime.now(),
                        'performance': best_metrics
                    }

                    updated_performance[product_id] = {
                        'arima': arima_metrics,
                        'exponential_smoothing': es_metrics,
                        'best_model': model_type,
                        'last_updated': datetime.now()
                    }

            except Exception as e:
                print(f"Error retraining model for product {product_id}: {e}")
                continue

        # Save updated models
        self._save_updated_models(updated_models, updated_performance)

        print(f"Retraining completed. Updated {len(updated_models)} product models.")
        return updated_models, updated_performance

    def _generate_time_series_data(self):
        """Generate synthetic time series data for training"""
        time_series_data = {}

        for _, product in self.products_data.iterrows():
            product_id = product['product_id']

            # Generate 365 days of historical data
            np.random.seed(int(product_id.replace('P', '')))  # Deterministic seed

            # Base demand with seasonal and trend components
            days = 365
            time_index = pd.date_range(end=datetime.now(), periods=days, freq='D')

            # Seasonal component (weekly pattern)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(days) / 7)

            # Trend component (slight upward trend)
            trend = 0.01 * np.arange(days)

            # Random noise
            noise = np.random.normal(0, 2, days)

            # Base demand from product data
            base_demand = product['daily_dispensing_avg']

            # Generate demand series
            demand = base_demand + seasonal + trend + noise
            demand = np.maximum(demand, 0)  # Ensure non-negative

            time_series_data[product_id] = pd.Series(demand, index=time_index)

        return time_series_data

    def _train_arima(self, time_series):
        """Train ARIMA model"""
        try:
            model = ARIMA(time_series, order=(1, 1, 1))
            fitted_model = model.fit()
            return fitted_model
        except:
            # Fallback to simple model
            return None

    def _train_exponential_smoothing(self, time_series):
        """Train Exponential Smoothing model"""
        try:
            model = ExponentialSmoothing(time_series, seasonal_periods=7, trend='add', seasonal='add')
            fitted_model = model.fit()
            return fitted_model
        except:
            # Fallback to simple model
            return None

    def _evaluate_model(self, model, time_series):
        """Evaluate model performance"""
        if model is None:
            return {'mae': float('inf'), 'rmse': float('inf'), 'aic': float('inf')}

        try:
            # Simple evaluation - in production would use proper train/test split
            predictions = model.fittedvalues
            actual = time_series

            mae = mean_absolute_error(actual, predictions)
            rmse = mean_squared_error(actual, predictions, squared=False)
            aic = model.aic if hasattr(model, 'aic') else float('inf')

            return {'mae': mae, 'rmse': rmse, 'aic': aic}
        except:
            return {'mae': float('inf'), 'rmse': float('inf'), 'aic': float('inf')}

    def _save_updated_models(self, models, performance):
        """Save updated models and performance metrics"""
        # Save models
        joblib.dump(models, 'pharmacy_time_series_models.joblib')

        # Save performance metrics
        joblib.dump(performance, 'pharmacy_model_performance.joblib')

        # Save products data
        joblib.dump(self.products_data, 'pharmacy_products_data.joblib')

        # Update training metadata
        metadata = {
            'last_training_date': datetime.now(),
            'n_products': len(models),
            'model_version': '2.0',
            'retraining_type': 'automated',
            'training_parameters': {
                'arima_order': (1, 1, 1),
                'es_seasonal_periods': 7,
                'es_trend': 'add',
                'es_seasonal': 'add'
            }
        }
        joblib.dump(metadata, 'training_metadata.joblib')

        print("Updated models and metadata saved successfully")

def main():
    """Main function for automated retraining"""
    retrainer = ModelRetrainer()

    if not retrainer.load_current_models():
        print("Failed to load current models. Starting fresh training.")
        # Force retraining
        models, performance = retrainer.retrain_models()
    else:
        # Check if retraining is needed
        if retrainer.check_retraining_needed():
            print("Retraining needed. Starting retraining process...")
            models, performance = retrainer.retrain_models()
        else:
            print("Retraining not needed. Models are up to date.")
            return

    # Print summary
    print(f"\nRetraining Summary:")
    print(f"- Products processed: {len(models)}")
    print(f"- Models updated: {len(models)}")
    print(f"- Training completed at: {datetime.now()}")

if __name__ == "__main__":
    main()