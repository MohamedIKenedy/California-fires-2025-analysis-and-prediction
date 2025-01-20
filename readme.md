# Weather-Enhanced Fire Predictor

## Overview

This system predicts wildfire spread patterns using weather data, historical fire perimeters, and machine learning. It generates interactive visualizations of current fire perimeters, predicted spread patterns, and model performance metrics.

## Visualizations

### Fire Prediction Map

The following map shows the current fire perimeter (red) and predicted spread over the next 24 hours (orange), taking into account weather conditions:

[View Weather Enhanced Fire Prediction](https://htmlpreview.github.io/MohamedIKenedy/California-fires-2025-analysis-and-prediction/blob/main/Results/weather_enhanced_fire_prediction.html)
### Model Performance Visualizations

#### Weather Forecast Analysis

Time series comparison of predicted vs actual fire danger indices:

```html
<iframe src="Results/weather_forecast_performance.html" width="100%" height="500px" frameborder="0"></iframe>
```

#### Weather Prediction Accuracy

Scatter plot showing the correlation between predicted and actual weather conditions:

```html
<iframe src="Results/weather_prediction_accuracy.html" width="100%" height="500px" frameborder="0"></iframe>
```

#### Fire Spread Prediction Accuracy

Analysis of the spread prediction model's performance:

```html
<iframe src="Results/spread_prediction_accuracy.html" width="100%" height="500px" frameborder="0"></iframe>
```

#### Feature Importance

Visualization of the most influential factors in spread prediction:

```html
<iframe src="Results/feature_importance.html" width="100%" height="500px" frameborder="0"></iframe>
```

## Usage Instructions

To view these visualizations:

1. Run the prediction model using the main script
2. The HTML files will be generated in your working directory
3. Open this README in a web browser to see the interactive visualizations
4. Each visualization can be interacted with independently:
   - Zoom and pan the fire prediction map
   - Hover over data points in the performance plots
   - Click legend items to toggle different data series
   - Use the time slider in the weather forecast visualization

## Notes

- The fire prediction map updates in real-time with new weather data
- All visualizations are interactive and support mouse/touch interactions
- The maps require an active internet connection for tile loading
- Performance metrics are updated with each model run

## Technical Requirements

- Modern web browser with JavaScript enabled
- Internet connection for map tiles
- Recommended minimum screen resolution: 1024x768
