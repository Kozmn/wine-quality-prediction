"""
Wine Quality Prediction API

A RESTful API service that predicts wine quality based on physicochemical properties
using a pre-trained XGBoost machine learning model. The API provides endpoints for
health checking, model information retrieval, and wine quality prediction with
automatic Swagger documentation.

"""

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Define root endpoint for API navigation
@app.route('/')
def home():
    """
    Root endpoint providing API overview and available endpoints.
    
    Returns:
        dict: Welcome message and endpoint navigation
    """
    return {
        "message": "Welcome to Wine Quality API!",
        "docs": "/docs/",
        "endpoints": {
            "health": "/health",
            "model_info": "/api/info", 
            "predict": "/api/predict"
        }
    }

# Configure Flask-RESTX for automatic API documentation
api = Api(app,
    title='Wine Quality API',
    description='Predicts wine quality based on physicochemical properties using XGBoost',
    doc='/docs/',
    prefix='/api'
)

# Load pre-trained machine learning models
try:
    model = joblib.load("../models/xgb_best_all_features.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None

# Define input schema for Swagger documentation and validation
wine_input = api.model('WineInput', {
    'fixed acidity': fields.Float(required=True, example=7.4, 
                                description='Tartaric acid concentration (g/dm³)'),
    'volatile acidity': fields.Float(required=True, example=0.7,
                                   description='Acetic acid concentration (g/dm³)'),
    'citric acid': fields.Float(required=True, example=0.0,
                              description='Citric acid concentration (g/dm³)'),
    'residual sugar': fields.Float(required=True, example=1.9,
                                 description='Residual sugar after fermentation (g/dm³)'),
    'chlorides': fields.Float(required=True, example=0.076,
                            description='Sodium chloride concentration (g/dm³)'),
    'free sulfur dioxide': fields.Float(required=True, example=11.0,
                                      description='Free SO2 concentration (mg/dm³)'),
    'total sulfur dioxide': fields.Float(required=True, example=34.0,
                                       description='Total SO2 concentration (mg/dm³)'),
    'density': fields.Float(required=True, example=0.9978,
                          description='Wine density (g/cm³)'),
    'pH': fields.Float(required=True, example=3.51,
                     description='pH level of the wine'),
    'sulphates': fields.Float(required=True, example=0.56,
                            description='Potassium sulphate concentration (g/dm³)'),
    'alcohol': fields.Float(required=True, example=9.4,
                          description='Alcohol content (% by volume)')
})

@api.route('/data')
class ApiData(Resource):
    """API metadata and status information endpoint"""
    
    def get(self):
        """
        Retrieve API metadata and current status.
        
        Returns:
            dict: API version, status, and model loading state
        """
        return {
            "message": "Wine Quality API",
            "version": "1.0",
            "status": "running",
            "model_loaded": model is not None
        }

@app.route('/health')
def health():
    """
    Health check endpoint for monitoring and load balancing.
    
    Returns:
        dict: System health status with timestamp and component states
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "server": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@api.route('/info')
class ModelInfo(Resource):
    """Machine learning model information endpoint"""
    
    def get(self):
        """
        Retrieve detailed information about the loaded ML model.
        
        Returns:
            dict: Model specifications, required features, and output description
            
        Raises:
            500: If model is not properly loaded
        """
        if model is None:
            api.abort(500, "Model not loaded")
        
        return {
            "model_type": "XGBoost Regressor",
            "features_count": 11,
            "required_features": [
                'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                'density', 'pH', 'sulphates', 'alcohol'
            ],
            "output_range": "3-8 (wine quality score)",
            "description": "Predicts wine quality based on physicochemical properties"
        }

@api.route('/predict')
class Predict(Resource):
    """Wine quality prediction endpoint"""
    
    @api.expect(wine_input)
    def post(self):
        """
        Predict wine quality based on physicochemical properties.
        
        Expects a JSON payload with all 11 required wine features.
        The model processes the input through feature scaling before prediction.
        
        Returns:
            dict: Predicted quality score, input features, and success message
            
        Raises:
            400: If required features are missing from input
            500: If model or scaler is not loaded
        """
        # Validate model availability
        if model is None or scaler is None:
            api.abort(500, "Model not loaded")
        
        # Extract input data from request
        data = api.payload
        
        # Define required feature names in correct order
        required_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        # Validate input completeness
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            api.abort(400, f"Missing features: {missing_features}")
        
        # Prepare features for prediction
        feature_values = [data[feature] for feature in required_features]
        features_array = np.array(feature_values).reshape(1, -1)
        
        # Apply feature scaling and make prediction
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        
        return {
            "predicted_quality": round(float(prediction), 2),
            "input_features": dict(zip(required_features, feature_values)),
            "message": "prediction successful"
        }


if __name__ == '__main__':
    """
    Application entry point for development server.
    
    Configures and starts the Flask development server with debug mode enabled.
    In production, use a proper WSGI server like Gunicorn or uWSGI.
    """
    print("Starting Flask app with Swagger...")
    print("API Documentation: http://localhost:8000/docs/")
    print("Health Check: http://localhost:8000/health")
    print("Prediction Endpoint: http://localhost:8000/api/predict")
    app.run(debug=True, port=8000)