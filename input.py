import streamlit as st
import pandas as pd
from loader import inventory_data


def app():
    st.sidebar.header("Input Parameters")

    # Product SKU Selection - Now with all 1200+ products
    product_options = [f"{row['product_id']} - {row['product_name']}" for _, row in inventory_data.iterrows()]
    selected_sku = st.sidebar.selectbox("Select Product SKU", product_options)

    # Extract product ID
    selected_product_id = selected_sku.split(' - ')[0]

    # Get product data
    product_data = inventory_data[inventory_data['product_id'] == selected_product_id].iloc[0]

    # Current Stock
    current_stock = st.sidebar.number_input(
        'Current Stock',
        min_value=0,
        max_value=50000,  # Increased max value to accommodate larger stock levels
        value=int(product_data['current_stock'])
    )

    # Lead Time
    lead_time = st.sidebar.number_input(
        'Lead Time (days)',
        min_value=1,
        max_value=30,
        value=int(product_data['supplier_lead_time_days'])
    )

    # Average Daily Usage
    avg_daily_usage = st.sidebar.number_input(
        'Average Daily Usage (units)',
        min_value=0.0,
        max_value=1000.0,
        value=float(product_data['daily_dispensing_avg']),
        format="%.1f"
    )

    # Daily usage history (last 30 days) - use actual data from dataset
    actual_usage = str(product_data['daily_usage_last_30_days'])
    daily_usage_history = st.sidebar.text_area(
        'Daily usage (last 30 days, comma-separated)',
        value=actual_usage,
        height=100
    )

    # Batch expiry date - use actual data from dataset
    actual_expiry = ','.join(product_data['batch_expiry_dates']) if isinstance(product_data['batch_expiry_dates'], list) else str(product_data['batch_expiry_dates'])
    batch_expiry = st.sidebar.text_input(
        'Batch expiry dates (comma-separated YYYY-MM-DD)',
        value=actual_expiry
    )

    st.sidebar.markdown('---')

    # Create comprehensive input data
    input_data = {
        'selected_sku': selected_sku,
        'product_id': selected_product_id,
        'current_stock': current_stock,
        'lead_time_days': lead_time,
        'avg_daily_usage': avg_daily_usage,
        'daily_usage_history': [float(x.strip()) for x in daily_usage_history.split(',') if x.strip()],
        'batch_expiry_dates': [x.strip() for x in batch_expiry.split(',') if x.strip()],
        'product_data': product_data
    }

    return input_data