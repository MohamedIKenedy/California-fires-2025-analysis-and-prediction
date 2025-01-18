import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union
import folium
from folium import plugins
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import logging  


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedFirePredictor:
    def __init__(self, csv_path, geojson_path, weather_path):
        self.csv_data = pd.read_csv(csv_path)
        self.geo_data = gpd.read_file(geojson_path)
        self.weather_data = pd.read_csv(weather_path)
        self.current_perimeter = None
        self.predicted_spread = None
        self.weather_models = {}
        self.spread_model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        """Enhanced preprocessing with additional features"""
        # Convert dates to datetime
        date_cols = ['poly_DateCurrent', 'FireDiscoveryDate', 'CreationDate', 'EditDate']
        for col in date_cols:
            self.csv_data[col] = pd.to_datetime(self.csv_data[col], errors='coerce')

        # Ensure consistent datetime format
        self.geo_data['poly_DateCurrent'] = pd.to_datetime(self.geo_data['poly_DateCurrent']).dt.tz_localize(None)
        self.weather_data['datetime'] = pd.to_datetime(self.weather_data['datetime']).dt.tz_localize(None)

        # Add temporal features
        self.weather_data['hour'] = self.weather_data['datetime'].dt.hour
        self.weather_data['month'] = self.weather_data['datetime'].dt.month
        self.weather_data['day_of_week'] = self.weather_data['datetime'].dt.dayofweek
        self.weather_data['season'] = self.weather_data['month'].apply(self._get_season)

        # Calculate enhanced fire danger index
        self.weather_data['fire_danger_index'] = self._calculate_fire_danger_index()
        
        # Calculate drought index (simplified)
        self.weather_data['drought_index'] = self._calculate_drought_index()
        
        # Add fuel moisture content estimation
        self.weather_data['fuel_moisture'] = self._estimate_fuel_moisture()
        
        # Resample weather data to hourly intervals and fill missing values
        self.weather_data = self.weather_data.set_index('datetime').resample('H').interpolate()
        
    def _get_season(self, month):
        """Convert month to season number"""
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Fall

    def _calculate_fire_danger_index(self):
        """Enhanced fire danger index calculation"""
        temp_factor = (self.weather_data['temp'] - 32) * 5/9  # Convert to Celsius
        rh_factor = (100 - self.weather_data['humidity']) / 100
        wind_factor = self.weather_data['windspeed'] * 1.60934  # Convert to km/h
        
        # Canadian Fire Weather Index (FWI) inspired calculation
        buildup_index = (temp_factor * 0.4) + (rh_factor * 0.4) + (wind_factor * 0.2)
        
        return (buildup_index * 100).clip(0, 100)

    def _calculate_drought_index(self):
        """Calculate simple drought index based on precipitation history"""
        # Use 30-day rolling window for precipitation
        rolling_precip = self.weather_data['precip'].rolling(window=30*24, min_periods=1).sum()
        max_precip = rolling_precip.max()
        
        # Normalize and invert (less precipitation = higher drought index)
        drought_index = (1 - (rolling_precip / max_precip)) * 100
        return drought_index

    def _estimate_fuel_moisture(self):
        """Estimate fuel moisture content based on weather conditions"""
        rh = self.weather_data['humidity']
        temp = self.weather_data['temp']
        precip = self.weather_data['precip']
        
        # Basic fuel moisture estimation
        base_moisture = rh * 0.5
        temp_effect = (130 - temp) * 0.3
        precip_effect = precip * 5
        
        fuel_moisture = (base_moisture + temp_effect + precip_effect).clip(0, 100)
        return fuel_moisture

    def train_weather_models(self):
            logging.info("Training weather models")
            weather_variables = ['temp', 'humidity', 'windspeed', 'fire_danger_index', 'fuel_moisture']
            
            for var in weather_variables:
                logging.info(f"Processing {var}")
                data = self.weather_data[var].dropna()
                
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(data, period=12)
                logging.info(f"Seasonal decomposition completed for {var}")
                
                # SARIMA parameters
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 24)
                
                # Train SARIMA model
                model = SARIMAX(
                    data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                self.weather_models[var] = model.fit(disp=False)
                logging.info(f"Trained SARIMA model for {var}")
            
            logging.info("Weather models training completed")

    def predict_weather(self, hours_ahead=24):
        """Predict weather conditions using SARIMA models"""
        forecasts = {}
        
        for var, model in self.weather_models.items():
            # Get forecast
            forecast = model.forecast(steps=hours_ahead)
            forecasts[var] = forecast
        
        # Combine forecasts into DataFrame
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.index = pd.date_range(
            start=self.weather_data.index[-1],
            periods=hours_ahead + 1,
            freq='H'
        )[1:]
        
        return forecast_df

    def train_spread_model(self):
        """Train enhanced XGBoost model for fire spread prediction"""
        # Merge fire and weather data
        combined_data = self._prepare_combined_data()
        
        # Prepare features
        feature_cols = [
            'temp', 'humidity', 'windspeed', 'winddir', 'cloudcover',
            'fire_danger_index', 'drought_index', 'fuel_moisture',
            'hour', 'month', 'day_of_week', 'season'
        ]
        
        X = combined_data[feature_cols]
        y = combined_data['area_acres']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost with improved parameters
        self.spread_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )
        
        self.spread_model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Calculate and print model performance
        train_pred = self.spread_model.predict(X_train_scaled)
        test_pred = self.spread_model.predict(X_test_scaled)
        
        print("Model Performance:")
        print(f"Train R2: {r2_score(y_train, train_pred):.4f}")
        print(f"Test R2: {r2_score(y_test, test_pred):.4f}")
        print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.4f}")

    def _prepare_combined_data(self):
        """Prepare combined dataset for model training"""
        # Merge fire and weather data
        combined_data = pd.merge_asof(
            self.geo_data.sort_values('poly_DateCurrent'),
            self.weather_data.reset_index().sort_values('datetime'),
            left_on='poly_DateCurrent',
            right_on='datetime',
            direction='backward'
        )
        
        # Drop rows with missing values
        combined_data = combined_data.dropna(subset=['area_acres'])
        
        return combined_data

    def predict_spread(self, current_geometry, weather_forecast, hours_ahead=24):
        """Enhanced fire spread prediction"""
        if isinstance(current_geometry, MultiPolygon):
            base_area = sum(polygon.area for polygon in current_geometry.geoms)
        else:
            base_area = current_geometry.area
        
        # Prepare feature matrix for prediction
        X_pred = weather_forecast[['temp', 'humidity', 'windspeed', 'fire_danger_index', 'fuel_moisture']]
        X_pred['hour'] = X_pred.index.hour
        X_pred['month'] = X_pred.index.month
        X_pred['day_of_week'] = X_pred.index.dayofweek
        X_pred['season'] = X_pred['month'].apply(self._get_season)
        
        # Add missing features
        X_pred['winddir'] = self.weather_data['winddir'].mean()  # Use average wind direction
        X_pred['cloudcover'] = self.weather_data['cloudcover'].mean()
        X_pred['drought_index'] = self.weather_data['drought_index'].iloc[-1]
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predict spread
        predicted_areas = self.spread_model.predict(X_pred_scaled)
        
        # Calculate final spread based on maximum predicted area
        max_predicted_area = max(predicted_areas)
        spread_factor = np.sqrt(max_predicted_area / base_area)
        
        # Create directional spread
        self.current_perimeter = current_geometry
        self.predicted_spread = self._create_directional_spread(
            current_geometry,
            spread_factor,
            weather_forecast['windspeed'].mean(),
            weather_forecast.index[-1]
        )
        
        return self.predicted_spread

    def _create_directional_spread(self, geometry, spread_factor, wind_speed, forecast_time):
        """Create enhanced directional spread prediction"""
        if isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
        else:
            polygons = [geometry]
            
        spread_polygons = []
        for poly in polygons:
            # Create wind-adjusted elliptical buffer
            center = poly.centroid
            angles = np.linspace(0, 360, 128)  # Increased resolution
            coords = []
            
            # Calculate time-based spread factor
            time_factor = min(1 + (wind_speed / 100), 2.0)
            
            for angle in angles:
                # Enhanced directional spread calculation
                wind_effect = np.cos(np.radians(angle)) * (wind_speed / 50)
                distance = poly.area * spread_factor * (1 + wind_effect) * time_factor
                
                # Calculate coordinate
                rad = np.radians(angle)
                x = center.x + distance * np.cos(rad)
                y = center.y + distance * np.sin(rad)
                coords.append((x, y))
                
            spread_polygons.append(Polygon(coords))
            
        return unary_union(spread_polygons)

    def create_visualization(self, center=None):
        """Create enhanced interactive map visualization"""
        if center is None:
            center = [
                self.current_perimeter.centroid.y,
                self.current_perimeter.centroid.x
            ]
        
        m = folium.Map(location=center, zoom_start=12,
                      tiles='CartoDB positron')
        
        # Add current fire perimeter
        folium.GeoJson(
            self.current_perimeter.__geo_interface__,
            name='Current Fire Perimeter',
            style_function=lambda x: {
                'fillColor': 'red',
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.5
            }
        ).add_to(m)
        
        # Add predicted spread with confidence intervals
        folium.GeoJson(
            self.predicted_spread.__geo_interface__,
            name='Predicted Spread (24h)',
            style_function=lambda x: {
                'fillColor': 'orange',
                'color': 'orange',
                'weight': 2,
                'fillOpacity': 0.3
            }
        ).add_to(m)
        
        # Add weather information with enhanced metrics
        latest_weather = self.weather_data.iloc[-1]
        weather_html = f"""
        <div style="position: fixed; top: 50px; right: 50px; z-index: 1000; 
                    background-color: white; padding: 10px; border-radius: 5px; 
                    border: 2px solid grey;">
            <h4>Current Conditions</h4>
            <p>Temperature: {latest_weather['temp']}°F</p>
            <p>Humidity: {latest_weather['humidity']}%</p>
            <p>Wind Speed: {latest_weather['windspeed']} mph</p>
            <p>Fire Danger Index: {latest_weather['fire_danger_index']:.1f}</p>
            <p>Drought Index: {latest_weather['drought_index']:.1f}</p>
            <p>Fuel Moisture: {latest_weather['fuel_moisture']:.1f}%</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(weather_html))
        
        # Add enhanced legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border-radius: 5px;
                    border: 2px solid grey;">
            <h4>Legend</h4>
            <p><span style="color: red;">■</span> Current Fire Perimeter</p>
            <p><span style="color: orange;">■</span> Predicted Spread (24h)</p>
            <p>Note: Prediction considers:</p>
            <ul style="padding-left: 20px;">
                <li>Weather conditions</li>
                <li>Drought index</li>
                <li>Fuel moisture</li>
                <li>Seasonal patterns</li>
            </ul>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add timeline of fire progression
        if hasattr(self, 'spread_timeline'):
            timeline_html = '''
            <div id="timeline" style="position: fixed; bottom: 50px; right: 50px; z-index: 1000;
                        background-color: white; padding: 10px; border-radius: 5px;
                        border: 2px solid grey; width: 200px;">
                <h4>Spread Timeline</h4>
                <div id="timeline-content"></div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(timeline_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

def main():
    """Main execution function with enhanced error handling and logging"""
    try:
        # Initialize predictor
        predictor = EnhancedFirePredictor(
            csv_path='CA_Perimeters_NIFC_FIRIS_public_view.csv',
            geojson_path='CA_Perimeters_NIFC_FIRIS_public_view.geojson',
            weather_path='california 2023-01-01 to 2025-01-18.csv'
        )
        
        print("Starting data preprocessing...")
        predictor.preprocess_data()
        
        print("Training weather prediction models...")
        predictor.train_weather_models()
        
        print("Training spread prediction model...")
        predictor.train_spread_model()
        
        # Get weather forecast
        print("Generating weather forecast...")
        forecast = predictor.predict_weather(hours_ahead=24)
        
        # Get the most recent fire perimeter
        latest_geometry = predictor.geo_data.iloc[-1].geometry
        
        print("Predicting fire spread...")
        predicted_spread = predictor.predict_spread(latest_geometry, forecast)
        
        print("Creating visualization...")
        map_viz = predictor.create_visualization()
        
        # Save visualization
        output_path = 'enhanced_fire_prediction.html'
        map_viz.save(output_path)
        print(f"Visualization saved to {output_path}")
        
        return map_viz, predictor
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    map_viz, predictor = main()