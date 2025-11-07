#!/usr/bin/env python3
"""
Daily Pharmacy Inventory Prediction Scheduler
Runs automated daily predictions and alerts for pharmacy inventory management.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class DailyInventoryScheduler:
    def __init__(self):
        self.models = None
        self.products_data = None
        self.performance_metrics = None
        self.load_models()

    def load_models(self):
        """Load trained models and data"""
        try:
            self.models = joblib.load('pharmacy_time_series_models.joblib')
            self.products_data = joblib.load('pharmacy_products_data.joblib')
            self.performance_metrics = joblib.load('pharmacy_model_performance.joblib')
            print(f"Loaded models for {len(self.models)} products")
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        return True

    def run_daily_predictions(self):
        """Run daily predictions for all products"""
        print(f"Starting daily predictions at {datetime.now()}")

        alerts = []
        predictions = {}

        for product_id in self.products_data['product_id'].values:
            product_data = self.products_data.loc[product_id]
            try:
                # Get product models
                if product_id in self.models:
                    model_data = self.models[product_id]

                    # Generate 30-day forecast
                    forecast = self.generate_forecast(product_id, days=30)
                    predictions[product_id] = forecast

                    # Check for alerts
                    product_alerts = self.check_alerts(product_id, forecast)
                    if product_alerts:
                        alerts.extend(product_alerts)

                else:
                    # New product - use fallback
                    fallback_alerts = self.handle_new_product(product_id)
                    if fallback_alerts:
                        alerts.extend(fallback_alerts)

            except Exception as e:
                print(f"Error processing product {product_id}: {e}")
                continue

        # Save daily results
        self.save_daily_results(predictions, alerts)

        print(f"Completed daily predictions. Generated {len(alerts)} alerts.")
        return predictions, alerts

    def generate_forecast(self, product_id, days=30):
        """Generate forecast for a product"""
        if product_id not in self.models:
            return None

        model_data = self.models[product_id]

        # Choose best model based on AIC
        arima_aic = self.performance_metrics.get(product_id, {}).get('arima', {}).get('aic', float('inf'))
        es_aic = self.performance_metrics.get(product_id, {}).get('exponential_smoothing', {}).get('aic', float('inf'))

        if arima_aic < es_aic and model_data.get('arima'):
            forecast = model_data['arima'].forecast(days)
        elif model_data.get('exponential_smoothing'):
            forecast = model_data['exponential_smoothing'].forecast(days)
        else:
            # Fallback to simple average
            historical = model_data.get('historical_data', [])
            avg_daily = np.mean(historical) if historical else 50
            forecast = np.full(days, avg_daily)

        return forecast

    def check_alerts(self, product_id, forecast):
        """Check for alerts based on predictions"""
        alerts = []
        product_data = self.products_data.loc[product_id]

        # Current stock and usage
        current_stock = product_data['current_stock']
        daily_usage = product_data['daily_dispensing_avg']
        lead_time = product_data['supplier_lead_time_days']

        # Calculate depletion date
        depletion_days = current_stock / daily_usage if daily_usage > 0 else 365
        depletion_date = datetime.now() + timedelta(days=depletion_days)

        # 1. Stockout Risk Alert
        if depletion_days <= lead_time:
            alerts.append({
                'product_id': product_id,
                'alert_type': 'STOCKOUT_RISK',
                'severity': 'HIGH',
                'message': f'Product {product_id} will stock out in {depletion_days:.0f} days, before reorder arrives',
                'action_required': 'Place emergency order immediately'
            })

        # 2. Low Stock Warning
        elif depletion_days <= 30:
            alerts.append({
                'product_id': product_id,
                'alert_type': 'LOW_STOCK',
                'severity': 'MEDIUM',
                'message': f'Product {product_id} has only {depletion_days:.0f} days of stock remaining',
                'action_required': 'Plan reorder within next 2 weeks'
            })

        # 3. Expiry Risk Alert (6 months rule) - Alert regardless of stockout risk
        batch_expiry_dates = product_data.get('batch_expiry_dates', [])
        if batch_expiry_dates:
            for i, expiry_str in enumerate(batch_expiry_dates):
                try:
                    expiry_date = pd.to_datetime(expiry_str)
                    days_to_expiry = (expiry_date - datetime.now()).days

                    # Alert if expiry within 6 months, regardless of stockout risk
                    if days_to_expiry <= 180:
                        severity = 'HIGH' if days_to_expiry <= 30 else 'MEDIUM' if days_to_expiry <= 90 else 'LOW'
                        alerts.append({
                            'product_id': product_id,
                            'alert_type': 'EXPIRY_RISK',
                            'severity': severity,
                            'message': f'Batch {i+1} of product {product_id} expires in {days_to_expiry} days ({expiry_date.strftime("%Y-%m-%d")})',
                            'action_required': 'Prioritize sales of expiring stock or arrange disposal'
                        })
                except:
                    continue

        # 4. Forecast Accuracy Alert
        if forecast is not None and len(forecast) > 0:
            # Check if forecast shows unusual patterns
            forecast_mean = np.mean(forecast)
            historical_mean = np.mean(product_data.get('daily_demand_history', [forecast_mean]))

            if abs(forecast_mean - historical_mean) / historical_mean > 0.5:  # 50% deviation
                alerts.append({
                    'product_id': product_id,
                    'alert_type': 'FORECAST_ANOMALY',
                    'severity': 'LOW',
                    'message': f'Product {product_id} shows unusual forecast pattern - review model accuracy',
                    'action_required': 'Monitor actual sales vs forecast'
                })

        return alerts

    def handle_new_product(self, product_id):
        """Handle predictions for new products without historical data"""
        alerts = []

        # Use category average (simplified - all products treated as similar)
        avg_daily_usage = self.products_data['daily_dispensing_avg'].mean()
        avg_lead_time = self.products_data['supplier_lead_time_days'].mean()

        alerts.append({
            'product_id': product_id,
            'alert_type': 'NEW_PRODUCT',
            'severity': 'INFO',
            'message': f'New product {product_id} - using category averages for predictions',
            'action_required': 'Monitor actual usage to build historical data'
        })

        return alerts

    def save_daily_results(self, predictions, alerts):
        """Save daily prediction results and alerts"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save predictions
        predictions_file = f'daily_predictions_{timestamp}.joblib'
        joblib.dump(predictions, predictions_file)

        # Save alerts
        alerts_df = pd.DataFrame(alerts)
        alerts_file = f'daily_alerts_{timestamp}.csv'
        alerts_df.to_csv(alerts_file, index=False)

        # Save summary
        summary = {
            'date': datetime.now(),
            'total_products': len(predictions),
            'total_alerts': len(alerts),
            'high_severity_alerts': len([a for a in alerts if a['severity'] == 'HIGH']),
            'predictions_file': predictions_file,
            'alerts_file': alerts_file
        }

        summary_file = f'daily_summary_{timestamp}.joblib'
        joblib.dump(summary, summary_file)

        print(f"Daily results saved: {summary_file}")

    def check_retraining_needed(self):
        """Check if model retraining is needed"""
        try:
            metadata = joblib.load('training_metadata.joblib')
            last_training = metadata['last_training_date']
            days_since_training = (datetime.now() - last_training).days

            # Retrain every 30 days or if performance degrades
            if days_since_training >= 30:
                print("Model retraining recommended (30+ days since last training)")
                return True

        except:
            print("Could not check training metadata")
            return False

        return False

def main():
    """Main function for daily scheduler"""
    scheduler = DailyInventoryScheduler()

    if not scheduler.load_models():
        print("Failed to load models. Exiting.")
        return

    # Run daily predictions
    predictions, alerts = scheduler.run_daily_predictions()

    # Check if retraining is needed
    if scheduler.check_retraining_needed():
        print("Consider running model retraining to maintain accuracy")

    # Print summary
    print(f"\nDaily Summary:")
    print(f"- Products processed: {len(predictions)}")
    print(f"- Alerts generated: {len(alerts)}")

    if alerts:
        print(f"- High severity alerts: {len([a for a in alerts if a['severity'] == 'HIGH'])}")
        print(f"- Medium severity alerts: {len([a for a in alerts if a['severity'] == 'MEDIUM'])}")
        print(f"- Low severity alerts: {len([a for a in alerts if a['severity'] == 'LOW'])}")

if __name__ == "__main__":
    main()