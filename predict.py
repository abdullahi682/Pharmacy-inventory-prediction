import time
import streamlit as st
from loader import model, accuracy_result
from data.config import thresholds
from function.function import make_donut
from data.base import mrk
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def app(input_data):
    # Extract data from input
    selected_sku = input_data['selected_sku']
    product_id = input_data['product_id']
    current_stock = input_data['current_stock']
    lead_time_days = input_data['lead_time_days']
    avg_daily_usage = input_data['avg_daily_usage']
    daily_usage_history = input_data['daily_usage_history']
    batch_expiry_dates = input_data['batch_expiry_dates']

    # Calculate forecasts using time series models if available
    try:
        # Load time series models
        time_series_models = joblib.load('pharmacy_time_series_models.joblib')
        if product_id in time_series_models and time_series_models[product_id]['arima'] is not None:
            # Use ARIMA model for forecasting
            arima_model = time_series_models[product_id]['arima']
            forecast_1w = float(arima_model.forecast(7).sum())
            forecast_1m = float(arima_model.forecast(30).sum())
            forecast_3m = float(arima_model.forecast(90).sum())
        elif product_id in time_series_models and time_series_models[product_id]['exponential_smoothing'] is not None:
            # Use Exponential Smoothing model
            es_model = time_series_models[product_id]['exponential_smoothing']
            forecast_1w = float(es_model.forecast(7).sum())
            forecast_1m = float(es_model.forecast(30).sum())
            forecast_3m = float(es_model.forecast(90).sum())
        else:
            # For new products without history, use category averages
            try:
                category_averages = joblib.load('pharmacy_category_averages.joblib')
                # Extract category from product name (assuming format: CategoryName_ProductID)
                product_name_parts = selected_sku.split('_')
                if len(product_name_parts) > 0:
                    category = product_name_parts[0].lower()
                    if category in category_averages:
                        # Use category average demand for forecasting
                        cat_avg = category_averages[category]['avg_daily_demand']
                        forecast_1w = cat_avg * 7
                        forecast_1m = cat_avg * 30
                        forecast_3m = cat_avg * 90
                        print(f"Using category averages for new product {product_id} (category: {category})")
                    else:
                        # Fallback to simple calculation
                        forecast_1w = avg_daily_usage * 7
                        forecast_1m = avg_daily_usage * 30
                        forecast_3m = avg_daily_usage * 90
                else:
                    # Fallback to simple calculation
                    forecast_1w = avg_daily_usage * 7
                    forecast_1m = avg_daily_usage * 30
                    forecast_3m = avg_daily_usage * 90
            except:
                # Fallback to simple calculation if category data not available
                forecast_1w = avg_daily_usage * 7
                forecast_1m = avg_daily_usage * 30
                forecast_3m = avg_daily_usage * 90
    except:
        # Fallback to simple calculation if models not available
        forecast_1w = avg_daily_usage * 7
        forecast_1m = avg_daily_usage * 30
        forecast_3m = avg_daily_usage * 90

    # Calculate depletion date
    depletion_date = datetime.now() + timedelta(days=int(current_stock / avg_daily_usage)) if avg_daily_usage > 0 else None

    # Enhanced expiry risk analysis with batch-level details
    expiry_risk = False
    expiry_alert_days = 180  # 6 months as specified
    expiry_will_expire_before_depletion = False
    batch_expiry_analysis = []
    total_stock_at_risk = 0

    if batch_expiry_dates:
        try:
            # Parse dates, handling potential semicolon separators
            parsed_dates = []
            for date_str in batch_expiry_dates:
                if date_str:
                    # Handle both comma and semicolon separators
                    dates = [d.strip() for d in date_str.replace(';', ',').split(',') if d.strip()]
                    for date in dates:
                        try:
                            parsed_dates.append(pd.to_datetime(date))
                        except:
                            continue

            if parsed_dates:
                # Sort dates for FEFO (First Expired, First Out) analysis
                sorted_dates = sorted(parsed_dates)

                # Analyze each batch
                cumulative_stock = 0
                for i, expiry_date in enumerate(sorted_dates):
                    days_to_expiry = (expiry_date - datetime.now()).days

                    # Calculate stock that would be consumed by this expiry date (FEFO logic)
                    stock_consumed_by_expiry = avg_daily_usage * max(0, days_to_expiry)
                    stock_remaining_at_expiry = max(0, current_stock - cumulative_stock - stock_consumed_by_expiry)

                    # Determine if this batch is at risk
                    batch_at_risk = days_to_expiry <= expiry_alert_days

                    # Check if significant stock from this batch will expire
                    if stock_remaining_at_expiry > current_stock * 0.05:  # 5% threshold
                        batch_at_risk = True
                        total_stock_at_risk += stock_remaining_at_expiry

                    batch_info = {
                        'batch_number': i + 1,
                        'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                        'days_to_expiry': days_to_expiry,
                        'stock_at_risk': stock_remaining_at_expiry,
                        'at_risk': batch_at_risk
                    }
                    batch_expiry_analysis.append(batch_info)

                    # Update cumulative stock for next batch
                    cumulative_stock += stock_remaining_at_expiry

                    # Overall expiry risk
                    if batch_at_risk:
                        expiry_risk = True

                # Check if any batch expires before total depletion
                if depletion_date and sorted_dates[0] < depletion_date:
                    expiry_will_expire_before_depletion = True
                    expiry_risk = True

        except Exception as e:
            print(f"Error parsing expiry dates: {e}")
            expiry_risk = False
            batch_expiry_analysis = []

    # Check stockout risk
    stock_days = current_stock / avg_daily_usage if avg_daily_usage > 0 else 0
    stockout_risk = stock_days < lead_time_days

    # Calculate stock days (how long current stock will last)
    stock_days = current_stock / avg_daily_usage if avg_daily_usage > 0 else 0

    # RISK ASSESSMENT: ONLY IF stock expected to last ≤ 180 days
    # System should only detect High, Medium, or Low risk IF AND ONLY IF
    # stock_days ≤ 180 days

    if stock_days <= 180:
        # Only assess risk if stock will last ≤ 180 days
        if stock_days <= 30:
            risk_level = "High Risk"
            risk_probability = 85  # High risk probability
        elif stock_days <= 90:
            risk_level = "Medium Risk"
            risk_probability = 65  # Medium risk probability
        else:  # 91-180 days
            risk_level = "Low Risk"
            risk_probability = 45  # Low risk probability
    else:
        # NO RISK ASSESSMENT if stock lasts more than 180 days
        risk_level = "No Risk"
        risk_probability = 0

    high_risk = risk_probability >= 50

    # Calculate last order date
    last_order_date = depletion_date - timedelta(days=lead_time_days) if depletion_date else None

    cols = st.columns(2)

    def stream_data():
        risk_status = "High Risk" if high_risk else "Low Risk"
        text = f"Model Accuracy: {accuracy_result}%\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        text = f"\nPrediction: {risk_level}\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        text = f"Risk Probability: {risk_probability:.1f}%\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        text = f"SKU: {product_id}\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        text = f"Forecast 1 Week: {forecast_1w:.1f} units\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        text = f"Forecast 1 Month: {forecast_1m:.1f} units\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        text = f"Forecast 3 Months: {forecast_3m:.1f} units\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        text = f"Avg Daily Usage: {avg_daily_usage:.1f} units\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        if depletion_date:
            text = f"Depletion Date: {depletion_date.strftime('%Y-%m-%d')}\n\n"
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.05)

        text = f"Expiry Risk: {'Yes' if expiry_risk else 'No'}"
        if expiry_will_expire_before_depletion:
            text += " (Stock will expire before depletion)"
        if total_stock_at_risk > 0:
            text += f" - {total_stock_at_risk:.0f} units at risk"
        text += "\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        # Show batch-level expiry details
        if batch_expiry_analysis:
            text = "Batch Expiry Details:\n"
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.05)

            for batch in batch_expiry_analysis:
                status = "⚠️ AT RISK" if batch['at_risk'] else "✅ Safe"
                text = f"Batch {batch['batch_number']}: {batch['expiry_date']} ({batch['days_to_expiry']} days) - {status}"
                if batch['stock_at_risk'] > 0:
                    text += f" - {batch['stock_at_risk']:.0f} units may expire"
                text += "\n"
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(0.05)
            text = "\n"
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.05)

        text = f"Stockout Risk: {'Yes' if stockout_risk else 'No'}\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)

        if last_order_date:
            text = f"Last Order Date: {last_order_date.strftime('%Y-%m-%d')}\n\n"
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.05)

        # Enhanced Decision Support Logic
        decision_text = f"Decision Support: Stock expected to last {int(stock_days)} days at current usage. "

        # Analyze batch expiry risks for decision making
        at_risk_batches = [batch for batch in batch_expiry_analysis if batch['at_risk']] if batch_expiry_analysis else []
        earliest_risk_batch = at_risk_batches[0] if at_risk_batches else None

        if high_risk:
            if expiry_risk and earliest_risk_batch:
                batch_num = earliest_risk_batch['batch_number']
                days_to_expiry = earliest_risk_batch['days_to_expiry']
                stock_at_risk = earliest_risk_batch['stock_at_risk']

                if days_to_expiry <= 90:  # Critical expiry risk
                    decision_text += f"Critical expiry risk for Batch {batch_num} ({days_to_expiry} days). Prioritize clearing {stock_at_risk:.0f} units immediately. Consider emergency redistribution or promotional pricing."
                elif days_to_expiry <= 180:  # Moderate expiry risk
                    decision_text += f"Expiry risk detected for Batch {batch_num} ({days_to_expiry} days). Adjust reordering strategy to prioritize depletion of Batch {batch_num} stock. Consider reducing next order quantity to clear the at-risk inventory before it expires."
                else:
                    decision_text += f"Expiry risk detected for Batch {batch_num}. Monitor closely and plan inventory rotation."

            elif stockout_risk:
                decision_text += f"Stockout risk detected. Lead time is {lead_time_days} days. Place reorder immediately to avoid stockouts. Consider increasing safety stock levels."
            else:
                decision_text += "Inventory levels require attention. Review usage patterns and consider strategic reordering."
        else:
            if expiry_risk and at_risk_batches:
                decision_text += f"Monitor expiry risk for {len(at_risk_batches)} batch(es). Continue normal operations but plan for batch rotation."
            else:
                decision_text += "Inventory levels are healthy. Continue normal monitoring and standard reordering procedures."

        decision_text += f" Monthly consumption: approximately {forecast_1m:.0f} units."

        # Add specific recommendations based on risk combinations
        if expiry_risk and stockout_risk:
            decision_text += " High priority: Address both expiry and stockout risks simultaneously."
        elif expiry_risk and not stockout_risk:
            decision_text += " Focus on inventory rotation to prevent expiry losses."
        elif not expiry_risk and stockout_risk:
            decision_text += " Prioritize replenishment to maintain service levels."

        decision_text += "\n"

        for word in decision_text.split(" "):
            yield word + " "
            time.sleep(0.05)

    cols[0].write_stream(stream_data)

    risk_status = risk_level
    color = "red" if risk_probability >= 70 else "orange" if risk_probability >= 40 else "blue"

    cols[1].markdown(mrk.format(color, risk_status), unsafe_allow_html=True)
    cols[1].write('\n\n\n\n\n')

    # Donut chart for risk probability - rounded to 1 decimal place
    donut_chart = make_donut(round(risk_probability, 1),
                            f'{risk_probability:.1f} %',
                            input_color=color)

    cols[1].altair_chart(donut_chart)