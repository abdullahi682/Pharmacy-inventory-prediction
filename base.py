st_style = """
           <style>
           #MainMenu {visibility: hidden;}
           footer {visibility: hidden;}
           header {visibility: hidden;}
           div.block-container {padding-top:1rem;}
           .css-ysnqb2 e1g8pov64 {margin-top: -75px;}
           </style>
           """

footer_pharmacy = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 255, 0.1);  /* Very transparent blue */
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #FFFFFF;  /* White text color */
    }
    .footer a {
        color: #FFD700;  /* Golden link color */
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <p>Pharmacy Inventory Prediction System | AI-powered inventory management | © 2025 AbdullahiAhm <a href="https://github.com/AbdullahiAhm/Pharmacy-Inventory-Prediction.git" target="_blank">GitHub</a></p>
    </div>
    """


head_pharmacy = """
    <div style="text-align:
    center;
    font-size: 40px;
    font-weight: bold;
    color: #2E86C1;
    margin-bottom: 20px;">
        🌟 Pharmacy Inventory Prediction App 🌟
    </div>
    <div style="text-align: center; font-size: 18px; color: #5D6D7E; margin-bottom: 60px;">
        Harness the power of time series forecasting to predict inventory needs and provide insights!
    </div>
    """

mrk = """
<div style="background-color: {}; 
color: white; 
margin-bottom: 50px;
padding: 10px;
max-width: 300px;
text-align: center;
border-radius: 5px; text-align: center;">
    {}
</div>
"""


about_pharmacy_management = """
## What is Pharmacy Inventory Management?

Pharmacy Inventory Management involves tracking and forecasting the stock levels of medications and medical supplies to ensure availability while minimizing waste from expired products. Effective management helps prevent stockouts and reduces costs associated with overstocking or expired goods.

### Key Concepts:
**Demand Forecasting:**

Predicting future usage based on historical dispensing data.
Uses time series models like Exponential Smoothing or ARIMA to account for trends and seasonality.

**Expiry Risk:**

Monitoring batches of medications with different expiry dates.
Alerts when stock may expire before being used.

**Stockout Risk:**

Predicting when current stock will be depleted.
Considering lead times for reordering from suppliers.

**Reorder Point:**

The optimal time to place an order to avoid stockouts.
Calculated based on usage rates and supplier lead times.

### Benefits of Inventory Prediction:
- Reduced stockouts leading to better patient care.
- Minimized waste from expired medications.
- Optimized ordering to reduce holding costs.
- Improved operational efficiency in pharmacies.

### Challenges:
- Variable demand patterns for different medications.
- Managing multiple batches with different expiry dates.
- Balancing cost savings with availability.

### Best Practices:
- **Regular Monitoring**: Daily checks of stock levels and expiry dates.
- **Data-Driven Decisions**: Use historical data for accurate forecasting.
- **Supplier Collaboration**: Maintain good relationships for reliable lead times.
- **Automation**: Implement systems for real-time tracking and alerts.
"""


warn_pharmacy_system = """
This project (forecasting system) was created for learning purposes, the predictions may have inaccuracies. Please use professional pharmacy management software for critical decisions.
"""