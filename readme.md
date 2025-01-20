# Advanced Wildfire Spread Prediction System Using Machine Learning and Environmental Data Analysis
<p align="center">
  <img src="https://raw.githubusercontent.com/MohamedIKenedy/California-fires-2025-analysis-and-prediction/main/Results/CA_Fires.png" alt="Model Output Visualization">
</p>

## Research Overview
This study presents a sophisticated predictive modeling system for wildfire behavior analysis, integrating multidimensional environmental data with machine learning algorithms to forecast fire spread patterns. The system employs a hybrid approach combining SARIMA models for weather prediction and XGBoost for spread dynamics.

## Methodology

### Data Sources
- **Fire Perimeter Data**: Historical California fire perimeters from NIFC and FIRIS
- **Meteorological Data**: High-resolution weather parameters (2023-2025)
- **Environmental Indices**: Derived fire danger and drought indices

### Model Architecture
1. **Weather Prediction Component**
   - SARIMA (Seasonal AutoRegressive Integrated Moving Average)
   - Temporal resolution: Hourly forecasts
   - Key parameters: Temperature, humidity, wind velocity, pressure gradients

2. **Fire Spread Prediction**
   - Algorithm: XGBoost Regressor
   - Feature engineering: 12 environmental and temporal variables
   - Optimization: Early stopping with cross-validation

### Evaluation Metrics
- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Cross-validation scores

## Results Visualization

### Interactive Fire Spread Analysis
Current perimeter analysis and 24-hour spread projection:

[View Dynamic Fire Prediction Model](https://mohamedkenedy.github.io/California-fires-2025-analysis-and-prediction/Results/weather_enhanced_fire_prediction.html)

![Model Output Visualization](https://raw.githubusercontent.com/MohamedIKenedy/California-fires-2025-analysis-and-prediction/main/Results/map_preview.png)

### Model Performance Analytics

#### Meteorological Forecast Validation
- Time series analysis of predicted vs observed fire danger indices
- Confidence intervals and error margins
- Seasonal decomposition metrics

View detailed analysis: [Weather Forecast Performance](Results/weather_forecast_performance.html)

#### Spread Prediction Accuracy
- Spatial accuracy assessment
- Temporal evolution analysis
- Error distribution patterns

View validation metrics: [Spread Prediction Accuracy](Results/spread_prediction_accuracy.html)

#### Variable Importance Analysis
Quantitative assessment of predictor variables' influence on model outcomes:
- Primary environmental factors
- Temporal dependencies
- Interaction effects

View analysis: [Feature Importance Analysis](Results/feature_importance.html)

## Technical Implementation

### System Requirements
- Python 3.8+
- Key Dependencies:
  - geopandas
  - xgboost
  - statsmodels
  - scikit-learn
  - folium

### Computational Parameters
- Training data temporal range: 2023-2025
- Spatial resolution: Variable based on fire perimeter data
- Forecast horizon: 24 hours
- Model update frequency: Hourly

### Usage Protocol
1. Initialize environmental variable configuration
2. Execute data preprocessing pipeline
3. Train prediction models
4. Generate visualization outputs

## Limitations and Future Work
- Current model assumes uniform fuel distribution
- Weather data temporal resolution constraints
- Computational efficiency in real-time predictions

## Acknowledgments
https://gis.data.cnra.ca.gov/datasets/CALFIRE-Forestry::ca-perimeters-nifc-firis-public-view

Note: This research is part of ongoing work in wildfire behavior prediction. Results should be interpreted within the context of model assumptions and limitations.
