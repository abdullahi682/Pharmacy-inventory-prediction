import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for pharmacy inventory data.
    Creates derived features for demand forecasting models.
    """
    def __init__(self):
        self.epsilon = 1e-5

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        # Pharmacy-specific features
        if 'daily_dispensing_avg' in data.columns and 'monthly_dispensing_avg' in data.columns:
            # Ratio of daily to monthly dispensing (normalized)
            data['DispensingRatio'] = data['daily_dispensing_avg'] / (data['monthly_dispensing_avg'] / 30 + self.epsilon)

        if 'current_stock' in data.columns and 'daily_dispensing_avg' in data.columns:
            # Number of days current stock will last
            data['StockDays'] = data['current_stock'] / (data['daily_dispensing_avg'] + self.epsilon)

        if 'supplier_lead_time_days' in data.columns and 'daily_dispensing_avg' in data.columns:
            # Safety stock calculation (lead time demand)
            data['LeadTimeDemand'] = data['supplier_lead_time_days'] * data['daily_dispensing_avg']

        if 'current_stock' in data.columns and 'supplier_lead_time_days' in data.columns and 'daily_dispensing_avg' in data.columns:
            # Stock coverage ratio (how many lead times current stock covers)
            lead_time_demand = data['supplier_lead_time_days'] * data['daily_dispensing_avg']
            data['StockCoverageRatio'] = data['current_stock'] / (lead_time_demand + self.epsilon)

        return data


class WoEEncoding(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) encoding transformer for categorical features.
    Converts continuous features into categorical bins and applies WoE transformation.
    """
    def __init__(self):
        # Pharmacy-specific feature bins for inventory management
        self.feature_bins = {
            'daily_dispensing_avg': [-np.inf, 25, 75, 150, 300, np.inf],
            'current_stock': [-np.inf, 500, 1500, 3000, 5000, np.inf],
            'StockDays': [-np.inf, 7, 14, 30, 60, np.inf],
            'supplier_lead_time_days': [-np.inf, 3, 7, 14, 30, np.inf],
            'DispensingRatio': [-np.inf, 0.8, 0.9, 1.0, 1.1, np.inf],
            'StockCoverageRatio': [-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        }
        self.woe_mappings = {}

    def fit(self, X, y):
        y = pd.Series(y, name='target')
        for feature, bins in self.feature_bins.items():
            if feature in X.columns:
                X_copy = X.copy()
                X_copy[f'{feature}_cat'] = pd.cut(X_copy[feature], bins=bins)
                woe_df = self._calculate_woe(X_copy, f'{feature}_cat', y)
                self.woe_mappings[feature] = woe_df.set_index(f'{feature}_cat')['WOE'].to_dict()
        return self

    def transform(self, X):
        data = X.copy()
        for feature in self.feature_bins.keys():
            if feature in data.columns:
                data[f'{feature}_cat'] = pd.cut(data[feature], bins=self.feature_bins[feature])
                data[f'{feature}_woe'] = data[f'{feature}_cat'].map(self.woe_mappings[feature])
                data.drop(columns=[f'{feature}_cat'], inplace=True)
        return data

    def _calculate_woe(self, data, feature_name, y):
        data_copy = data.copy()
        data_copy['target'] = y
        grouped = data_copy.groupby(feature_name, observed=False)['target'].value_counts().unstack(fill_value=0)

        # Handle case where we might not have both classes
        if grouped.shape[1] < 2:
            grouped['non_events'] = grouped.get(0, 0)
            grouped['events'] = grouped.get(1, 0)
        else:
            grouped.columns = ['non_events', 'events']

        # Calculate WoE with smoothing to avoid division by zero
        total_events = grouped['events'].sum()
        total_non_events = grouped['non_events'].sum()

        if total_events == 0:
            grouped['WOE'] = -10  # Very negative WoE for no events
        elif total_non_events == 0:
            grouped['WOE'] = 10   # Very positive WoE for no non-events
        else:
            grouped['event_rate'] = (grouped['events'] + 0.5) / (total_events + 1)  # Laplace smoothing
            grouped['non_event_rate'] = (grouped['non_events'] + 0.5) / (total_non_events + 1)
            grouped['WOE'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])

        return grouped.reset_index()
    

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Column selector transformer for pharmacy inventory features.
    Selects specified columns from the dataset, handling missing columns gracefully.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Select only columns that exist in the data
        available_columns = [col for col in self.columns if col in X.columns]
        if not available_columns:
            # If no specified columns exist, return all columns
            return X
        return X[available_columns]