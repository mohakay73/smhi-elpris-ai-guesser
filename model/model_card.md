---
language: sv
license: mit
tags:
- electricity-price
- regression
- smhi
- nordpool
- se2
---

# SMHI Electricity Price Predictor — SE2

Day-ahead machine learning model predicting electricity prices in price area **SE2** using historical Nord Pool spot prices and true SMHI weather forecasts (`stg__forecast`).

## Architecture
- **Model**: `HistGradientBoostingRegressor`
- **Features**: Lagged prices (`price_today`, `price_yesterday`, `price_7d_ago`, `price_7d_mean`), calendar features, and true day-ahead weather forecasts (`tomorrow_temp_mean`, `tomorrow_wind_mean`, etc.).
- **Data Source**: Neon PostgreSQL (`stg__elpris`, `stg__weather`, `stg__forecast`).