# Standard library imports
import asyncio
import base64
import calendar
import html
import io
import json
import logging
import os
import re
import ssl
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Tuple

# Third-party imports
import chardet
import featuretools as ft
import google.generativeai as genai
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pycountry
from prophet import Prophet
import scipy
from scipy import stats
import seaborn as sns
import shap
import spacy
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from textblob import TextBlob

# Flask and related imports
from flask import (
    Flask, flash, jsonify, make_response, redirect, render_template, 
    request, send_file, session, url_for
)
from flask.json import JSONEncoder
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Scikit-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error, 
    mean_squared_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, PolynomialFeatures, PowerTransformer, StandardScaler
)

# Tensorflow/Keras imports
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Other library imports
from currency_converter import CurrencyConverter
from fuzzywuzzy import fuzz

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle





# Initialize Flask app and extensions
app = Flask(__name__, static_folder='static')
app.secret_key = 'd823243f4b3906be44452a1c8f82ee0a'
socketio = SocketIO(app, json=json, cors_allowed_origins="*")
jwt = JWTManager(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
STATIC_FOLDER = 'static'
VISUALIZATIONS_FOLDER = 'visualizations'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['VISUALIZATIONS_FOLDER'] = VISUALIZATIONS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'

# Gemini API Configuration
API_KEY = 'AIzaSyCsSVU10C_pPWxT3mtsLOdKqYA0qMvFMjU'
genai.configure(api_key=API_KEY)

# Create necessary folders
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, STATIC_FOLDER, VISUALIZATIONS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Setup logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
mlflow.set_experiment("Advanced Data Cleaning Pipeline")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize currency converter
c = CurrencyConverter()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)




@dataclass
class ColumnMetadata:
    name: str
    dtype: str
    missing_count: int
    unique_count: int
    mean: float = None
    median: float = None
    std: float = None
    min_value: Any = None
    max_value: Any = None



def save_data_to_disk(session_id, data):
    joblib.dump(data, f'data_{session_id}.joblib')

def load_data_from_disk(session_id):
    try:
        return joblib.load(f'data_{session_id}.joblib')
    except:
        return None


class DataManager:
    def __init__(self):
        self._data = {}
        self._original_data = None
        self._feature_store = {}
        self._transformed_data = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)


    def set_transformed_data(self, session_id, data):
        with self._lock:
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Attempted to set invalid transformed data type")
                raise ValueError("Transformed data must be a pandas DataFrame")
            self._transformed_data[session_id] = data.copy()
            self.logger.info(f"Transformed data set in DataManager for session {session_id}. Shape: {data.shape}")

    def get_transformed_data(self, session_id):
        with self._lock:
            if session_id not in self._transformed_data:
                self.logger.warning(f"No transformed data found for session {session_id}")
                return None
            return self._transformed_data[session_id].copy()





    def set_data(self, session_id, data):
        with self._lock:
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Attempted to set invalid data type")
                raise ValueError("Data must be a pandas DataFrame")
            self._data[session_id] = data.copy()
            self.logger.info(f"Data set in DataManager for session {session_id}. Shape: {data.shape}")

    def get_data(self, session_id):
        with self._lock:
            if session_id not in self._data:
                self.logger.warning(f"No data found for session {session_id}")
                return None
            return self._data[session_id].copy()

    def update_data(self, session_id, new_data):
        with self._lock:
            if not isinstance(new_data, pd.DataFrame):
                self.logger.error("Attempted to update with invalid data type")
                raise ValueError("New data must be a pandas DataFrame")
            self._data[session_id] = new_data.copy()
            self.logger.info(f"Data updated in DataManager for session {session_id}. New shape: {new_data.shape}")






    def reset_data(self):
        with self._lock:
            if self._original_data is None:
                logging.warning("Attempted to reset data, but no original data is set.")
            else:
                self._data = self._original_data.copy()
                logging.info("Data reset to original state.")

    def add_feature(self, feature_name, feature_data):
        self._feature_store[feature_name] = feature_data

    def get_feature(self, feature_name):
        return self._feature_store.get(feature_name)

    def is_data_available(self, session_id):
        with self._lock:
            return session_id in self._data and not self._data[session_id].empty

    def clear_data(self, session_id):
        with self._lock:
            if session_id in self._data:
                del self._data[session_id]
            if session_id in self._transformed_data:
                del self._transformed_data[session_id]
            self.logger.info(f"Data cleared for session {session_id}")



# Create an instance of DataManager
data_manager = DataManager()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class UniversalAIML:
    def __init__(self, df):
        self.df = df
        self.target_column = None
        self.feature_columns = None
        self.is_classification = None
        self.is_time_series = None
        self.model = None
        self.session_id = None  # Add this line



    def set_session_id(self, session_id):
        self.session_id = session_id



    def identify_target(self, target_column):
        self.target_column = target_column
        self.feature_columns = [col for col in self.df.columns if col != self.target_column]
        
        # Determine if it's a classification or regression problem
        unique_values = self.df[self.target_column].nunique()
        self.is_classification = unique_values < 10  # Arbitrary threshold

        # Check if it's a time series problem
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        self.is_time_series = len(date_columns) > 0

    def preprocess_data(self):
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = X[col].astype('category').cat.codes

        return X, y

    def train_model(self):
        if self.is_time_series:
            return self.train_prophet_model()
        elif self.is_classification:
            return self.train_classification_model()
        else:
            return self.train_regression_model()

    def train_prophet_model(self):
        date_column = self.df.select_dtypes(include=['datetime64']).columns[0]
        df_prophet = pd.DataFrame({
            'ds': self.df[date_column],
            'y': self.df[self.target_column]
        })
        model = Prophet()
        model.fit(df_prophet)
        self.model = model
        return "Prophet model trained successfully"

    def train_classification_model(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.model = model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }

    def train_regression_model(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.model = model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }

    def make_predictions(self, input_data):
        if self.is_time_series:
            future = self.model.make_future_dataframe(periods=len(input_data))
            forecast = self.model.predict(future)
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(input_data))
            return predictions.to_dict('records')
        else:
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            input_data = input_data[self.feature_columns]
            
            # Handle categorical variables
            categorical_columns = input_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                le.fit(self.df[col].astype(str))
                input_data[col] = le.transform(input_data[col].astype(str))

            predictions = self.model.predict(input_data)
            
            if isinstance(predictions, np.ndarray):
                predictions = [{'predicted': float(p)} for p in predictions]
            elif isinstance(predictions, pd.Series):
                predictions = [{'predicted': float(p)} for p in predictions]
            elif isinstance(predictions, pd.DataFrame):
                predictions = predictions.to_dict('records')
            
            return predictions

    def evaluate_model(self):
        results = {}  # Initialize results dictionary
        try:
            if self.is_time_series:
                # For time series, we'll use the last 20% of the data for testing
                train_size = int(0.8 * len(self.df))
                train = self.df[:train_size]
                test = self.df[train_size:]
                
                date_column = self.df.select_dtypes(include=['datetime64']).columns[0]
                df_prophet = pd.DataFrame({
                    'ds': train[date_column],
                    'y': train[self.target_column]
                })
                model = Prophet()
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)
                
                mse = mean_squared_error(test[self.target_column], forecast['yhat'][-len(test):])
                mae = mean_absolute_error(test[self.target_column], forecast['yhat'][-len(test):])
                r2 = r2_score(test[self.target_column], forecast['yhat'][-len(test):])
                
                sample_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).to_dict('records')
                sample_predictions = [
                    {
                        'true': float(test[self.target_column].iloc[i]) if i < len(test) else None,
                        'predicted': float(pred['yhat'])
                    }
                    for i, pred in enumerate(sample_predictions)
                ]
            else:
                X, y = self.preprocess_data()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = self.model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                sample_predictions = [
                    {'true': float(true), 'predicted': float(pred)}
                    for true, pred in zip(y_test[:10], y_pred[:10])
                ]

            results = {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "sample_predictions": [
                    {
                        'true': float(pred['true']) if pred['true'] is not None and not pd.isna(pred['true']) else None,
                        'predicted': float(pred['predicted']) if pred['predicted'] is not None and not pd.isna(pred['predicted']) else None
                    }
                    for pred in sample_predictions
                ]
            }
        except Exception as e:
            logging.error(f"Error in evaluate_model: {str(e)}")
            results = {
                "error": f"An error occurred during model evaluation: {str(e)}",
                "mse": None,
                "mae": None,
                "r2": None,
                "sample_predictions": []
            }
        
        return results

class AdvancedDataPipeline:
    def __init__(self, df):
        self.df = df
        self.original_df = df.copy()
        self.metadata = None
        self.universal_aiml = UniversalAIML(df)
        self.logger = logging.getLogger(__name__)
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns






    def update_column_types(self):
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns

    def update_data(self, new_df):
        self.df = new_df
        self.update_column_types()
        self.universal_aiml.df = new_df




    def get_full_data_summary(self):
        self.update_column_types()  # Ensure we have the latest column classifications
        summary = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'numeric_columns': self.numeric_cols.tolist(),
            'categorical_columns': self.categorical_cols.tolist(),
            'datetime_columns': self.datetime_cols.tolist(),
            'column_details': {}
        }
        
        for col in self.df.columns:
            col_summary = {
                'dtype': str(self.df[col].dtype),
                'unique_count': int(self.df[col].nunique()),
                'missing_count': int(self.df[col].isnull().sum()),
                'sample_values': self.df[col].sample(min(5, len(self.df))).tolist()
            }
            
            if col in self.numeric_cols:
                col_summary.update({
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max())
                })
            elif col in self.categorical_cols:
                col_summary['top_categories'] = {str(k): int(v) for k, v in self.df[col].value_counts().nlargest(5).to_dict().items()}
            
            summary['column_details'][col] = col_summary
        
        return summary








    def generate_metadata(self):
        try:
            self.metadata = []
            for col in self.df.columns:
                col_data = self.df[col]
                meta = ColumnMetadata(
                    name=col,
                    dtype=str(col_data.dtype),
                    missing_count=col_data.isnull().sum(),
                    unique_count=col_data.nunique()
                    )
                if pd.api.types.is_numeric_dtype(col_data):
                    meta.mean = col_data.mean()
                    meta.median = col_data.median()
                    meta.std = col_data.std()
                    meta.min_value = col_data.min()
                    meta.max_value = col_data.max()
                self.metadata.append(meta)
        except Exception as e:
            logger.error(f"Error in generate_metadata: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def handle_inconsistent_formats(self):
        try:
            logger.info("Handling inconsistent formats...")
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Strip whitespace
                    self.df[col] = self.df[col].str.strip()
                    
                    # Remove special characters except for commas, periods, and spaces
                    self.df[col] = self.df[col].str.replace(r'[^\w\s,.]', '', regex=True)
                    
                    # Remove extra spaces
                    self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
            return self.df, None, "Handled inconsistent formats by removing special characters (except commas and periods) from text columns."
        except Exception as e:
            logger.error(f"Error in handle_inconsistent_formats: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error handling inconsistent formats: {str(e)}", None

    def handle_missing_values(self):
        try:
            for column in self.df.columns:
                if self.df[column].dtype == object:
                    self.df[column] = self.df[column].fillna('').astype(str)
                elif pd.api.types.is_numeric_dtype(self.df[column]):
                    self.df[column] = self.df[column].fillna(self.df[column].median()).round(2)
                elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0] if not self.df[column].mode().empty else pd.NaT)
                elif self.df[column].dtype == bool:
                    self.df[column] = self.df[column].fillna(False)
            return self.df, None, "Handled missing values in all columns."
        except Exception as e:
            logger.error(f"Error in handle_missing_values: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error handling missing values: {str(e)}", None

    def handle_outliers(self):
        try:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df[column] = self.df[column].clip(lower_bound, upper_bound).round(2)
            return self.df, None, f"Handled outliers in {len(numeric_columns)} numeric columns."
        except Exception as e:
            logger.error(f"Error in handle_outliers: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error handling outliers: {str(e)}", None

    def handle_duplicates(self, similarity_threshold=80):
        try:
            logger.info("Handling duplicates including near-duplicates...")
            initial_rows = len(self.df)
            
            # Drop exact duplicates
            self.df.drop_duplicates(inplace=True)
            
            # Check for near-duplicates only if there are enough rows to make it meaningful
            if len(self.df) < 2:
                return self.df, None, "Not enough data to check for near-duplicates after removing exact duplicates."
            
            # Identify and remove near-duplicates
            def calculate_similarity(row1, row2):
                return np.mean([fuzz.ratio(str(a), str(b)) for a, b in zip(row1, row2)])
            
            rows_to_remove = set()
            for i in range(len(self.df)):
                if i in rows_to_remove:
                    continue
                for j in range(i + 1, len(self.df)):
                    if j in rows_to_remove:
                        continue
                    similarity = calculate_similarity(self.df.iloc[i], self.df.iloc[j])
                    if similarity > similarity_threshold:
                        rows_to_remove.add(j)
            
            self.df = self.df.drop(index=rows_to_remove)
            
            rows_removed = initial_rows - len(self.df)
            logger.info(f"Removed {rows_removed} duplicate and near-duplicate rows")

            return self.df, None, f"Removed {rows_removed} duplicate and near-duplicate rows."
        except Exception as e:
            logger.error(f"Error in handle_duplicates: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error handling duplicates: {str(e)}", None

    def handle_high_dimensionality(self):
        try:
            logger.info("Handling high dimensionality...")
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) <= 1:
                return self.df, None, "Not enough numeric columns for dimensionality reduction."

            pca = PCA(n_components=0.95)
            pca_result = pca.fit_transform(self.df[numeric_columns])
            pca_df = pd.DataFrame(data=pca_result, columns=[f'PC_{i+1}' for i in range(pca_result.shape[1])])

            mi_scores = mutual_info_regression(self.df[numeric_columns], self.df[self.df.columns[-1]])
            mi_scores = pd.Series(mi_scores, name="MI Scores", index=numeric_columns)
            top_features = mi_scores.nlargest(10).index.tolist()

            self.df = pd.concat([self.df, pca_df], axis=1)
            self.df = self.df[[col for col in self.df.columns if col in top_features or col.startswith('PC_')]]

            return self.df, None, f"Reduced dimensions from {len(numeric_columns)} to {len(self.df.columns)} columns."
        except Exception as e:
            logger.error(f"Error in handle_high_dimensionality: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error handling high dimensionality: {str(e)}", None

    def scale_and_transform_data(self):
        try:
            logger.info("Scaling and transforming data...")
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) == 0:
                return self.df, None, "No numeric columns to scale and transform."

            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            self.df[numeric_columns] = pt.fit_transform(self.df[numeric_columns])

            return self.df, None, f"Scaled and transformed {len(numeric_columns)} numeric columns."
        except Exception as e:
            logger.error(f"Error in scale_and_transform_data: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error scaling and transforming data: {str(e)}", None

    def feature_engineering(self):
        try:
            logger.info("Performing feature engineering...")
            es = ft.EntitySet(id="data")
            es = es.add_dataframe(dataframe_name="data", dataframe=self.df, index="index" if "index" in self.df.columns else None)
            
            feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="data",
                                                  trans_primitives=["add_numeric", "multiply_numeric"])

            target = self.df.columns[-1]
            mi_scores = mutual_info_regression(feature_matrix.drop(columns=[target]), feature_matrix[target])
            mi_scores = pd.Series(mi_scores, name="MI Scores", index=feature_matrix.columns)
            mi_scores = mi_scores.sort_values(ascending=False)
            top_features = mi_scores[mi_scores > mi_scores.mean()].index.tolist()

            self.df = feature_matrix[top_features + [target]]
            
            return self.df, None, f"Engineered {len(self.df.columns) - 1} new features."
        except Exception as e:
            logger.error(f"Error in feature_engineering: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error in feature engineering: {str(e)}", None

    def dynamic_feature_transformation(self):
        try:
            logger.info("Performing dynamic feature transformation...")
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            target = self.df.columns[-1]

            if len(numeric_columns) <= 1:
                return self.df, None, "Not enough numeric columns for dynamic feature transformation."

            best_features = set(numeric_columns)

            for i in range(3):  # Limit to 3 iterations to prevent excessive computation
                new_features = set()
                for col1 in best_features:
                    for col2 in best_features:
                        if col1 != col2:
                            self.df[f'{col1}_plus_{col2}'] = self.df[col1] + self.df[col2]
                            new_features.add(f'{col1}_plus_{col2}')
                            self.df[f'{col1}_times_{col2}'] = self.df[col1] * self.df[col2]
                            new_features.add(f'{col1}_times_{col2}')

                all_features = list(best_features.union(new_features))
                mi_scores = mutual_info_regression(self.df[all_features], self.df[target])
                mi_scores = pd.Series(mi_scores, index=all_features).sort_values(ascending=False)

                keep_features = mi_scores.index[:len(mi_scores)//2]
                best_features = set(keep_features)

                for feature in new_features:
                    if feature not in best_features:
                        del self.df[feature]

            return self.df, None, f"Created and selected {len(best_features) - len(numeric_columns)} new dynamic features."
        except Exception as e:
            logger.error(f"Error in dynamic_feature_transformation: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error in dynamic feature transformation: {str(e)}", None

    def advanced_data_transformation(self):
        try:
            logger.info("Performing advanced data transformation...")
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) == 0:
                return self.df, None, "No numeric columns for advanced transformation."

            # Handle missing values before transformation
            imputer = SimpleImputer(strategy='median')
            df_imputed = pd.DataFrame(imputer.fit_transform(self.df[numeric_columns]), columns=numeric_columns)

            # Apply log transformation to positive columns
            for col in numeric_columns:
                if df_imputed[col].min() > 0:
                    df_imputed[f'{col}_log'] = np.log(df_imputed[col])

            # Apply polynomial features
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df_imputed)

            # Get feature names (compatible with both older and newer scikit-learn versions)
            if hasattr(poly, 'get_feature_names_out'):
                feature_names = poly.get_feature_names_out(df_imputed.columns)
            else:
                feature_names = poly.get_feature_names(df_imputed.columns)

            poly_features = pd.DataFrame(poly_features, columns=feature_names)

            # Combine original and new features
            result_df = pd.concat([self.df, poly_features], axis=1)

            # Generate interaction terms
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    result_df[f'{col1}_interact_{col2}'] = self.df[col1] * self.df[col2]

            new_columns = set(result_df.columns) - set(self.df.columns)
            return result_df, None, f"Created {len(new_columns)} new features through advanced transformation."
        except Exception as e:
            logger.error(f"Error in advanced_data_transformation: {str(e)}")
            logger.error(traceback.format_exc())
            return self.df, f"Error in advanced data transformation: {str(e)}", None

    def build_ml_model(self, target_column):
        try:
            logging.info("Building advanced machine learning model...")
            self.universal_aiml.identify_target(target_column)
            results = self.universal_aiml.train_model()
            logging.info(f"Model training results: {results}")
            return results
        except Exception as e:
            logging.error(f"Error in build_ml_model: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def run_pipeline(self, target_column):
        try:
            with mlflow.start_run():
                self.generate_metadata()
                self.df, error, message = self.handle_inconsistent_formats()
                if error:
                    return None, error
                self.df = auto_detect_and_convert_dtypes(self.df)
                self.df, error, message = self.handle_missing_values()
                if error:
                    return None, error
                self.df, error, message = self.handle_outliers()
                if error:
                    return None, error
                self.df, error, message = self.handle_duplicates()
                if error:
                    return None, error
                self.df, error, message = self.handle_high_dimensionality()
                if error:
                    return None, error
                self.df, error, message = handle_categorical_data(self.df)
                if error:
                    return None, error
                self.df, error, message = self.scale_and_transform_data()
                if error:
                    return None, error
                self.df, error, message = self.feature_engineering()
                if error:
                    return None, error
                self.df, error, message = self.dynamic_feature_transformation()
                if error:
                    return None, error
                self.df, error, message = self.advanced_data_transformation()
                if error:
                    return None, error
                model_results = self.build_ml_model(target_column)

                # Log metrics and artifacts
                mlflow.log_dict(model_results, "model_results.json")
                mlflow.sklearn.log_model(self.universal_aiml.model, "model")

                logger.info("Pipeline execution complete")
                return model_results, None
        except MlflowException as e:
            logger.error(f"MLflow error: {str(e)}")
            # Proceed with the pipeline without MLflow logging
            return self.run_pipeline_without_mlflow(target_column)
        except Exception as e:
            logger.error(f"Unexpected error in pipeline execution: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    


    def generate_chat_visualizations(self, user_input):
        visualizations = []
        
        # Generate a histogram for a mentioned numeric column
        for col in self.numeric_cols:
            if col.lower() in user_input.lower():
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=self.df[col],
                    nbinsx=30,
                    name=col,
                    marker_color='rgba(0, 128, 255, 0.7)',
                    opacity=0.8
                ))
                fig.update_layout(
                    title=f"Distribution of {col}",
                    xaxis_title=col,
                    yaxis_title="Frequency",
                    bargap=0.1,
                    template="plotly_white"
                )
                visualizations.append(('histogram', fig.to_json()))
        
        # Generate a bar plot for a mentioned categorical column
        for col in self.categorical_cols:
            if col.lower() in user_input.lower():
                value_counts = self.df[col].value_counts().nlargest(10)  # Top 10 categories
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color='rgba(50, 171, 96, 0.7)',
                    name=col
                ))
                fig.update_layout(
                    title=f"Top 10 Categories in {col}",
                    xaxis_title=col,
                    yaxis_title="Count",
                    bargap=0.2,
                    template="plotly_white"
                )
                visualizations.append(('bar', fig.to_json()))
        
        # Generate a line plot for a datetime column and a numeric column
        if len(self.datetime_cols) > 0 and len(self.numeric_cols) > 0:
            datetime_col = self.datetime_cols[0]
            numeric_col = self.numeric_cols[0]
            if datetime_col.lower() in user_input.lower() or numeric_col.lower() in user_input.lower():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.df[datetime_col],
                    y=self.df[numeric_col],
                    mode='lines+markers',
                    name=numeric_col,
                    line=dict(color='rgba(255, 165, 0, 0.8)', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title=f"{numeric_col} over time",
                    xaxis_title=datetime_col,
                    yaxis_title=numeric_col,
                    template="plotly_white"
                )
                visualizations.append(('line', fig.to_json()))
        
        # Generate a scatter plot for two mentioned numeric columns
        mentioned_numeric_cols = [col for col in self.numeric_cols if col.lower() in user_input.lower()]
        if len(mentioned_numeric_cols) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.df[mentioned_numeric_cols[0]],
                y=self.df[mentioned_numeric_cols[1]],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.df[mentioned_numeric_cols[1]],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Data Points'
            ))
            fig.update_layout(
                title=f"{mentioned_numeric_cols[1]} vs {mentioned_numeric_cols[0]}",
                xaxis_title=mentioned_numeric_cols[0],
                yaxis_title=mentioned_numeric_cols[1],
                template="plotly_white"
            )
            visualizations.append(('scatter', fig.to_json()))
        
        # Generate a heatmap for correlation between numeric columns
        if "correlation" in user_input.lower() or "heatmap" in user_input.lower():
            corr_matrix = self.df[self.numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmin=-1, zmax=1
            ))
            fig.update_layout(
                title="Correlation Heatmap",
                template="plotly_white",
                height=600,
                width=800
            )
            visualizations.append(('heatmap', fig.to_json()))
        
        # Generate a box plot for a mentioned numeric column grouped by a categorical column
        if "box plot" in user_input.lower() or "boxplot" in user_input.lower():
            numeric_col = next((col for col in self.numeric_cols if col.lower() in user_input.lower()), None)
            categorical_col = next((col for col in self.categorical_cols if col.lower() in user_input.lower()), None)
            if numeric_col and categorical_col:
                fig = go.Figure()
                for category in self.df[categorical_col].unique():
                    fig.add_trace(go.Box(
                        y=self.df[self.df[categorical_col] == category][numeric_col],
                        name=str(category),
                        boxpoints='outliers'
                    ))
                fig.update_layout(
                    title=f"Box Plot of {numeric_col} by {categorical_col}",
                    yaxis_title=numeric_col,
                    template="plotly_white"
                )
                visualizations.append(('boxplot', fig.to_json()))
        
        # Generate a pie chart for a mentioned categorical column
        if "pie chart" in user_input.lower() or "piechart" in user_input.lower():
            categorical_col = next((col for col in self.categorical_cols if col.lower() in user_input.lower()), None)
            if categorical_col:
                value_counts = self.df[categorical_col].value_counts().nlargest(10)  # Top 10 categories
                fig = go.Figure(data=[go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    hole=.3,
                    textinfo='percent+label'
                )])
                fig.update_layout(
                    title=f"Pie Chart of Top 10 Categories in {categorical_col}",
                    template="plotly_white"
                )
                visualizations.append(('piechart', fig.to_json()))
        
        return visualizations














    def run_pipeline_without_mlflow(self, target_column):
        try:
            logger.info("Running pipeline without MLflow logging...")
            self.generate_metadata()
            self.df, error, message = self.handle_inconsistent_formats()
            if error:
                return None, error
            self.df = auto_detect_and_convert_dtypes(self.df)
            self.df, error, message = self.handle_missing_values()
            if error:
                return None, error
            self.df, error, message = self.handle_outliers()
            if error:
                return None, error
            self.df, error, message = self.handle_duplicates()
            if error:
                return None, error
            self.df, error, message = self.handle_high_dimensionality()
            if error:
                return None, error
            self.df, error, message = handle_categorical_data(self.df)
            if error:
                return None, error
            self.df, error, message = self.scale_and_transform_data()
            if error:
                return None, error
            self.df, error, message = self.feature_engineering()
            if error:
                return None, error
            self.df, error, message = self.dynamic_feature_transformation()
            if error:
                return None, error
            self.df, error, message = self.advanced_data_transformation()
            if error:
                return None, error
            model_results = self.build_ml_model(target_column)

            logger.info("Pipeline execution complete (without MLflow logging)")
            return model_results, None
        except Exception as e:
            logger.error(f"Error in run_pipeline_without_mlflow: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_basic_insights(self):
        if self.df is None:
            return "No data available. Please upload data first."

        try:
            insights = []

            # Basic dataset information
            insights.append(f"Dataset shape: {self.df.shape}")
            insights.append(f"Columns: {', '.join(self.df.columns)}")

            # Missing values
            missing_data = self.df.isnull().sum()
            insights.append("\nMissing values:")
            for column, count in missing_data[missing_data > 0].items():
                insights.append(f"  {column}: {count} ({count/len(self.df):.2%})")

            # Numeric column statistics
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            insights.append("\nNumeric column statistics:")
            for col in numeric_cols:
                insights.append(f"  {col}:")
                insights.append(f"    Mean: {self.df[col].mean():.2f}")
                insights.append(f"    Median: {self.df[col].median():.2f}")
                insights.append(f"    Std Dev: {self.df[col].std():.2f}")
                insights.append(f"    Min: {self.df[col].min():.2f}")
                insights.append(f"    Max: {self.df[col].max():.2f}")

            # Categorical column information
            cat_cols = self.df.select_dtypes(include=['object']).columns
            insights.append("\nCategorical column information:")
            for col in cat_cols:
                insights.append(f"  {col}:")
                insights.append(f"    Unique values: {self.df[col].nunique()}")
                insights.append(f"    Top 3 values: {', '.join(self.df[col].value_counts().nlargest(3).index.astype(str))}")

            # Correlation insights
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                high_corr = corr_matrix[abs(corr_matrix) > 0.7].stack().reset_index()
                high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
                if not high_corr.empty:
                    insights.append("\nHigh correlations:")
                    for _, row in high_corr.iterrows():
                        insights.append(f"  {row['level_0']} and {row['level_1']}: {row[0]:.2f}")

            # Data types
            insights.append("\nColumn data types:")
            for col, dtype in self.df.dtypes.items():
                insights.append(f"  {col}: {dtype}")

            # Sample data
            insights.append("\nSample data (first 5 rows):")
            insights.append(self.df.head().to_string())

            return "\n".join(insights)
        except Exception as e:
            return f"Error in generate_basic_insights: {str(e)}"

    def generate_ai_insights(self, basic_insights):
        if self.df is None:
            return "No data available. Please upload data first."

        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""Analyze the following dataset summary and provide key insights, patterns, and recommendations:

                Dataset Summary:
                {basic_insights}

                Please provide:
                1. Key insights about the data
                2. Potential patterns or trends
                3. Recommendations for further analysis or actions
                4. Any potential issues or areas of concern in the data
                5. Suggestions for feature engineering or model selection based on the data characteristics
                """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating AI insights: {str(e)}")
            return f"Error generating AI insights: {str(e)}"

    def generate_insights(self):
        basic_insights = self.generate_basic_insights()
        ai_insights = self.generate_ai_insights(basic_insights)
        return basic_insights + "\n\n" + ai_insights


    def chat_with_data(self, user_input):
        try:
            model = genai.GenerativeModel('gemini-pro')
            
            data_summary = f"""
            Dataset Summary:
            - Shape: {self.df.shape}
            - Columns: {', '.join(self.df.columns)}
            - Numeric columns: {', '.join(self.df.select_dtypes(include=[np.number]).columns)}
            - Non-numeric columns: {', '.join(self.df.select_dtypes(exclude=[np.number]).columns)}
            """
            
            prompt = f"""
            {data_summary}
            
            User Query: {user_input}
            
            Based on this dataset and the user's query, provide a detailed analysis or answer.
            Consider relationships between variables, statistical trends, and real-world implications when answering.
            If appropriate, suggest a visualization that would help illustrate the answer.
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error in chat_with_data: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"



    @staticmethod
    def json_serializer(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

        





    def generate_visualizations(self, max_cols=25):
        try:
            logging.info("Starting advanced visualization generation process...")
            visualizations = []
            attempted = []

            logging.info(f"Column counts - Numeric: {len(self.numeric_cols)}, Categorical: {len(self.categorical_cols)}, Datetime: {len(self.datetime_cols)}")

            def safe_create_viz(viz_type, viz_func, *args):
                attempted.append(viz_type)
                try:
                    result = viz_func(*args)
                    if result is not None:
                        visualizations.append((viz_type, result))
                        logging.info(f"Successfully created {viz_type} visualization")
                    else:
                        logging.warning(f"Failed to create {viz_type} visualization: result was None")
                except Exception as viz_error:
                    logging.error(f"Error creating {viz_type} visualization: {str(viz_error)}")
                    logging.error(traceback.format_exc())

            # Data overview
            safe_create_viz('data_overview', self.create_data_overview)

            # Numeric visualizations
            if len(self.numeric_cols) > 0:
                safe_create_viz('numeric_summary', self.create_numeric_summaries, self.numeric_cols[:max_cols])
                safe_create_viz('correlation_heatmap', self.create_correlation_heatmap, self.numeric_cols[:max_cols])
                
                if len(self.numeric_cols) >= 2:
                    safe_create_viz('scatter_matrix', self.create_scatter_matrix, self.numeric_cols[:min(4, max_cols)])
                
                if len(self.numeric_cols) >= 3:
                    safe_create_viz('3d_scatter', self.create_3d_scatter, self.numeric_cols[:3])
                
                safe_create_viz('parallel_coordinates', self.create_parallel_coordinates, self.numeric_cols[:max_cols])
                safe_create_viz('pca_plot', self.create_pca_plot, self.numeric_cols[:max_cols])
                safe_create_viz('cluster_plot', self.create_cluster_plot, self.numeric_cols[:2])
                safe_create_viz('anomaly_detection', self.create_anomaly_detection_plot, self.numeric_cols[:2])

            # Categorical visualizations
            if len(self.categorical_cols) > 0:
                safe_create_viz('categorical_summary', self.create_categorical_summaries, self.categorical_cols[:max_cols])
                
                if len(self.categorical_cols) >= 2:
                    safe_create_viz('categorical_sunburst', self.create_categorical_sunburst, self.categorical_cols[:2])

            # Time series visualization
            if len(self.datetime_cols) > 0 and len(self.numeric_cols) > 0:
                safe_create_viz('time_series', self.create_time_series_plot, self.datetime_cols[0], self.numeric_cols[0])

            logging.info(f"Visualization generation complete. Attempted {len(attempted)}, successful {len(visualizations)}")
            
            return visualizations, attempted

        except Exception as e:
            logging.error(f"Unexpected error in generate_visualizations: {str(e)}")
            logging.error(traceback.format_exc())
            return [('error', f"Error generating visualizations: {str(e)}")], ['error']

    def create_data_overview(self):
        summary = self.df.describe(include='all').T
        summary['missing'] = self.df.isnull().sum()
        summary['missing_percent'] = (self.df.isnull().sum() / len(self.df)) * 100

        fig = go.Figure(data=[go.Table(
            header=dict(values=['Column', 'Type', 'Non-Empty Count', 'Empty Count', 'Empty %', 'Unique Values'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[
                summary.index,
                self.df.dtypes,
                self.df.count(),
                summary['missing'],
                summary['missing_percent'].round(2),
                self.df.nunique()
            ],
            fill_color='lavender',
            align='left')
        )])
        fig.update_layout(
            title='Dataset Overview',
            height=400 + (len(self.df.columns) * 25),
            margin=dict(l=20, r=20, t=60, b=20),
            autosize=True
        )
        return fig

    def create_numeric_summaries(self, columns):
        fig = make_subplots(rows=len(columns), cols=2, 
                            subplot_titles=[f"{col} Distribution" for col in columns for _ in range(2)])
        for i, col in enumerate(columns, 1):
            fig.add_trace(go.Histogram(x=self.df[col], name=f"{col} Histogram", nbinsx=30), row=i, col=1)
            fig.add_trace(go.Box(y=self.df[col], name=f"{col} Box Plot"), row=i, col=2)
        fig.update_layout(
            height=300*len(columns),
            title_text="Numeric Column Distributions",
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60),
            autosize=True
        )
        fig.update_xaxes(title_text="Value")
        fig.update_yaxes(title_text="Frequency", col=1)
        fig.update_yaxes(title_text="Value Distribution", col=2)
        return fig

    def create_correlation_heatmap(self, columns):
        corr_matrix = self.df[columns].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(
            title="Correlation Between Numeric Variables",
            height=600,
            width=800,
            xaxis_title="Variables",
            yaxis_title="Variables",
            margin=dict(l=60, r=60, t=60, b=60),
            autosize=True
        )
        return fig

    def create_categorical_summaries(self, columns, top_n=15):
        fig = make_subplots(
            rows=len(columns), cols=1, 
            vertical_spacing=0.2  # Increase vertical spacing between subplots
        )
        for i, col in enumerate(columns, 1):
            value_counts = self.df[col].value_counts()
            top_n_counts = value_counts.nlargest(top_n)
            other_count = value_counts.sum() - top_n_counts.sum()
            if other_count > 0:
                top_n_counts['Other'] = other_count
            fig.add_trace(go.Bar(x=top_n_counts.index, y=top_n_counts.values, name=col), row=i, col=1)
            
            # Update x-axis for each subplot individually
            fig.update_xaxes(tickangle=-45, title_text=col, automargin=True, row=i, col=1)
        
        fig.update_layout(
            height=500 * len(columns),  # Adjust height per subplot
            title_text="Top 15 Categories for Each Categorical Variable",
            showlegend=False,
            margin=dict(l=80, r=20, t=100, b=50, pad=4),
        )
        
        fig.update_yaxes(title_text="Count", title_standoff=15)
        
        # Adjust layout for better spacing
        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            bargap=0.15,
            bargroupgap=0.1
        )
        
        return fig
        





    def create_time_series_plot(self, date_column, value_column):
        fig = go.Figure(data=go.Scatter(x=self.df[date_column], y=self.df[value_column], mode='lines+markers'))
        fig.update_layout(
            title=f"{value_column} Trend Over Time",
            xaxis_title="Date",
            yaxis_title=value_column,
            height=500,
            margin=dict(l=60, r=60, t=60, b=60),
            autosize=True
        )
        return fig

    def create_scatter_matrix(self, columns):
        fig = px.scatter_matrix(
            self.df[columns],
            dimensions=columns,
            title="Relationships Between Numeric Variables"
        )
        fig.update_layout(
            height=800,
            width=800,
            margin=dict(l=60, r=60, t=60, b=60),
            autosize=True
        )
        fig.update_traces(diagonal_visible=False)
        return fig

    def create_3d_scatter(self, columns):
        if len(columns) >= 3:
            fig = px.scatter_3d(
                self.df, x=columns[0], y=columns[1], z=columns[2],
                color=self.df[columns[0]], 
                title="3D Relationship Visualization"
            )
            fig.update_layout(
                height=700,
                margin=dict(l=60, r=60, t=60, b=60),
                autosize=True
            )
            return fig
        return None

    def create_parallel_coordinates(self, columns, limit=10):
        top_n = self.df.nlargest(limit, columns[0])
        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = top_n[columns[0]],
                            colorscale = 'Viridis',
                            showscale = True,
                            colorbar = dict(title = columns[0])),
                dimensions = [dict(label=col, values=top_n[col]) for col in columns]
            )
        )
        fig.update_layout(
            title=f'Multi-Dimensional Comparison (Top {limit} by {columns[0]})',
            height=600,
            margin=dict(l=80, r=80, t=80, b=80),
            autosize=True
        )
        return fig

    def create_pca_plot(self, columns):
        if len(columns) > 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.df[columns])
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            variance_explained = pca.explained_variance_ratio_ * 100
            fig = px.scatter(
                x=pca_result[:, 0], y=pca_result[:, 1],
                title="Principal Component Analysis (PCA)",
                labels={'x': f'PC1 ({variance_explained[0]:.2f}% explained)',
                        'y': f'PC2 ({variance_explained[1]:.2f}% explained)'}
            )
            fig.update_layout(
                height=600,
                margin=dict(l=60, r=60, t=60, b=60),
                autosize=True
            )
            return fig
        return None

    def create_cluster_plot(self, columns):
        if len(columns) >= 2:
            kmeans = KMeans(n_clusters=3)
            clusters = kmeans.fit_predict(self.df[columns])
            fig = px.scatter(
                self.df, x=columns[0], y=columns[1],
                color=clusters, 
                title="Data Clustering Visualization",
                labels={columns[0]: columns[0], columns[1]: columns[1]},
                color_continuous_scale=px.colors.qualitative.Set1
            )
            fig.update_layout(
                height=600,
                margin=dict(l=60, r=60, t=60, b=60),
                autosize=True
            )
            return fig
        return None

    def create_categorical_sunburst(self, columns):
        if len(columns) >= 2:
            parent_col, child_col = columns[:2]
            df_grouped = self.df.groupby([parent_col, child_col]).size().reset_index(name='count')
            df_grouped = df_grouped.nlargest(25, 'count')
            fig = px.sunburst(
                df_grouped, 
                path=[parent_col, child_col], 
                values='count',
                title=f"Hierarchical View of {parent_col} and {child_col}"
            )
            fig.update_layout(
                height=700,
                margin=dict(l=0, r=0, t=60, b=0),
                autosize=True
            )
            return fig
        return None

    def create_anomaly_detection_plot(self, columns):
        if len(columns) >= 2:
            X = self.df[columns]
            iso = IsolationForest(contamination=0.1)
            outliers = iso.fit_predict(X) == -1
            fig = px.scatter(
                self.df, x=columns[0], y=columns[1],
                color=outliers, 
                title="Anomaly Detection in Data",
                labels={columns[0]: columns[0], columns[1]: columns[1]},
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            fig.update_layout(
                height=600,
                margin=dict(l=60, r=60, t=60, b=60),
                autosize=True,
                legend_title_text='Is Anomaly'
            )
            return fig
        return None





# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(file_path):
    try:
        file_extension = file_path.rsplit('.', 1)[1].lower()
        logging.info(f"Attempting to read file: {file_path}")
        logging.info(f"File extension: {file_extension}")

        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        logging.info(f"Detected file encoding: {file_encoding}")

        if file_extension in ['csv', 'txt']:
            delimiters = [',', '\t', ';', '|']
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, sep=delimiter, encoding=file_encoding,
                                     low_memory=False, keep_default_na=False, na_values=[])
                    if df.shape[1] > 1:
                        logging.info(f"Successfully read file with delimiter: '{delimiter}'")
                        break
                except Exception as e:
                    logging.warning(f"Failed to read with delimiter '{delimiter}': {str(e)}")
            else:
                df = pd.read_csv(file_path, header=None, encoding=file_encoding,
                                 low_memory=False, keep_default_na=False, na_values=[])
                if df.shape[1] == 1:
                    df = df[0].str.split(expand=True)
                    logging.info("File read as a single column and then split")
                else:
                    raise ValueError("Could not determine the correct delimiter")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, keep_default_na=False, na_values=[])
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if df.empty:
            raise ValueError("The file is empty")

        logging.info(f"File read successfully. Shape: {df.shape}")

        df.columns = df.columns.astype(str).str.strip()

        logging.info(f"First few rows of the dataframe:\n{df.head().to_string()}")

        return df

    except Exception as e:
        logging.error(f"Error reading file: {str(e)}", exc_info=True)
        raise ValueError(f"Could not read the file. Error: {str(e)}")

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)

# Use this encoder for the Flask app
app.json_encoder = CustomJSONEncoder

def safe_serialize(obj):
    return json.loads(json.dumps(obj, cls=CustomJSONEncoder))

def prepare_data_for_json(data):
    def safe_convert(val):
        if pd.isna(val) or val is pd.NaT:
            return None
        if isinstance(val, (np.int64, np.float64)):
            return float(val)
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return str(val)

    return [{key: safe_convert(value) for key, value in row.items()} for row in data]

def identify_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()

def encode_categorical_columns(df, columns):
    try:
        logging.info(f"Encoding categorical columns: {columns}")
        new_columns = []
        warning_message = ""
        for column in columns:
            if column not in df.columns:
                warning_message += f"Column '{column}' not found in dataframe. "
                logging.warning(warning_message)
                continue

            logging.info(f"Processing column: {column}")
            logging.info(f"Column dtype: {df[column].dtype}")
            unique_values = df[column].unique()
            logging.info(f"Unique values: {unique_values}")

            if len(unique_values) > 5:
                warning_message += f"Column '{column}' has more than 5 categories ({len(unique_values)}). This may mess up the data. "
                logging.warning(warning_message)
                continue

            le = LabelEncoder()
            df[f'{column}_encoded'] = le.fit_transform(df[column].astype(str))

            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logging.info(f"Encoding mapping for {column}: {mapping}")

            new_columns.append(f'{column}_encoded')

        logging.info(f"Final columns after encoding: {df.columns.tolist()}")
        logging.info(f"New columns created: {new_columns}")
        return df, new_columns, None, warning_message.strip()
    except Exception as e:
        logging.error(f"Error in encode_categorical_columns: {str(e)}", exc_info=True)
        return df, [], None, f"Error encoding categorical columns: {str(e)}"

def handle_datetime_column(df, column):
    try:
        logging.info(f"Processing datetime column: {column}")
        new_columns = []
        warning_message = ""
        if column not in df.columns:
            warning_message += f"Column '{column}' not found in dataframe. "
            logging.warning(warning_message)
            return df, new_columns, None, warning_message.strip()

        logging.info(f"Column dtype: {df[column].dtype}")
        sample = df[column].dropna().astype(str).sample(min(1000, len(df[column]))).tolist()
        date_count = sum(is_date(x) for x in sample)
        logging.info(f"Date-like values: {date_count} out of {len(sample)} sampled")

        if date_count / len(sample) < 0.95:
            warning_message += f"Column '{column}' does not appear to contain primarily datetime values. "
            logging.warning(warning_message)
            return df, new_columns, None, warning_message.strip()

        df[column] = pd.to_datetime(df[column], errors='coerce')
        non_datetime = df[column].isna().sum()
        logging.info(f"Non-convertible values: {non_datetime}")

        if df[column].notna().sum() > 0:
            df[f'{column}_year'] = df[column].dt.year
            df[f'{column}_month'] = df[column].dt.month
            df[f'{column}_day'] = df[column].dt.day
            df[f'{column}_weekday'] = df[column].dt.weekday.map(lambda x: calendar.day_name[x])
            new_columns = [f'{column}_year', f'{column}_month', f'{column}_day', f'{column}_weekday']

            if (df[column].dt.hour != 0).any() or (df[column].dt.minute != 0).any():
                df[f'{column}_hour'] = df[column].dt.hour
                df[f'{column}_minute'] = df[column].dt.minute
                new_columns.extend([f'{column}_hour', f'{column}_minute'])

        logging.info(f"New columns created: {new_columns}")
        logging.info(f"Final columns after datetime processing: {df.columns.tolist()}")

        if non_datetime > 0:
            warning_message += f"{non_datetime} values in column '{column}' could not be converted to datetime. "

        return df, new_columns, None, warning_message.strip()

    except Exception as e:
        logging.error(f"Error processing datetime column '{column}': {str(e)}", exc_info=True)
        return df, [], None, f"Error processing datetime column '{column}': {str(e)}"

def handle_currency_column(df, column):
    try:
        logging.info(f"Processing column for number extraction: {column}")
        new_columns = []
        warning_message = ""
        
        if column not in df.columns:
            warning_message += f"Column '{column}' not found in dataframe. "
            logging.warning(warning_message)
            return df, new_columns, None, warning_message.strip()

        logging.info(f"Column dtype: {df[column].dtype}")
        unique_values = df[column].unique()
        logging.info(f"Unique values sample: {unique_values[:5]}")  # Log first 5 unique values

        if pd.api.types.is_numeric_dtype(df[column]):
            warning_message += f"Column '{column}' is already numeric. "
            logging.warning(warning_message)
            return df, new_columns, None, warning_message.strip()

        def extract_numeric(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float)):
                return float(value)
            value = str(value)
            numbers = re.findall(r'-?\d+(?:\.\d+)?', value)
            if numbers:
                return float(numbers[-1])  # Return the last number found
            return np.nan

        temp_series = df[column].apply(extract_numeric)

        non_numeric = temp_series.isna().sum()
        numeric_count = temp_series.notna().sum()
        logging.info(f"Numeric values found: {numeric_count}")
        logging.info(f"Non-numeric or null values: {non_numeric}")

        if numeric_count > 0:
            new_column_name = f'{column}_numeric'
            df[new_column_name] = temp_series
            new_columns.append(new_column_name)
            logging.info(f"New column created: {new_column_name}")
            logging.info(f"Sample of new column: {df[new_column_name].head()}")
            if non_numeric > 0:
                warning_message += f"{non_numeric} values in column '{column}' could not be converted to numeric. "
        else:
            warning_message += f"No numeric values found to extract in column '{column}'. "
            logging.warning(warning_message)

        return df, new_columns, None, warning_message.strip()

    except Exception as e:
        logging.error(f"Error processing column '{column}' for number extraction: {str(e)}", exc_info=True)
        return df, [], None, f"Error processing column '{column}': {str(e)}"

def handle_address_column(df, column):
    try:
        logging.info(f"Processing address column: {column}")
        new_columns = []
        warning_message = ""
        if column not in df.columns:
            warning_message += f"Column '{column}' not found in dataframe. "
            logging.warning(warning_message)
            return df, new_columns, None, warning_message.strip()

        logging.info(f"Column dtype: {df[column].dtype}")
        df[f'{column}_parsed'] = df[column].apply(parse_address_custom)
        for component in ['street', 'city', 'state', 'postcode', 'country']:
            df[f'{column}_{component}'] = df[f'{column}_parsed'].apply(lambda x: x.get(component, ''))
            new_columns.append(f'{column}_{component}')
        df = df.drop(columns=[f'{column}_parsed'])
        
        logging.info(f"New columns created: {new_columns}")
        logging.info(f"Final columns after address processing: {df.columns.tolist()}")

        if not new_columns:
            warning_message += f"No address components could be extracted from column '{column}'. "
            logging.warning(warning_message)
        
        return df, new_columns, None, warning_message.strip()
    except Exception as e:
        logging.error(f"Error processing address column '{column}': {str(e)}", exc_info=True)
        return df, [], None, f"Error processing address column '{column}': {str(e)}"

def parse_address_custom(address):
    components = re.split(r',\s*', str(address))
    parsed = {
        "street": "", "city": "", "state": "", "postcode": "", "country": ""
    }
    if len(components) >= 1:
        parsed["street"] = components[0]
    if len(components) >= 2:
        parsed["city"] = components[1]
    if len(components) >= 3:
        state_postcode = components[2].split()
        if len(state_postcode) > 1:
            parsed["state"] = ' '.join(state_postcode[:-1])
            parsed["postcode"] = state_postcode[-1]
        else:
            parsed["state"] = components[2]
    if len(components) >= 4:
        parsed["country"] = components[3]
    return parsed

def is_date(string):
    if not isinstance(string, str):
        return False

    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}',
        r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}',
        r'\d{2,4}\s+[A-Za-z]{3,9}\s+\d{1,2}',
        r'\d{1,2}:\d{2}(:\d{2})?(\s*[APap][Mm])?',
        r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?(\s*[APap][Mm])?',
        r'\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(:\d{2})?(\s*[APap][Mm])?',
        r'\d{1,2}-\d{1,2}-\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?(\s*[APap][Mm])?',
        r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?(\s*[APap][Mm])?',
        r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?(\s*[APap][Mm])?',
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?',
        r'^[1-9]\d{8,10}$'
    ]
    
    return any(re.match(pattern, string.strip()) for pattern in date_patterns)

def emit_progress(message):
    print(f"Emitting progress: {message}")
    socketio.emit('transformation_progress', {'message': message})
    socketio.sleep(0)

def handle_categorical_data(df):
    try:
        logger.info("Handling categorical data...")
        categorical_columns = df.select_dtypes(include=['object']).columns

        low_cardinality = [col for col in categorical_columns if df[col].nunique() < 10]
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[low_cardinality])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(low_cardinality))

        high_cardinality = [col for col in categorical_columns if df[col].nunique() >= 10]
        for col in high_cardinality:
            hashed_features = pd.util.hash_pandas_object(df[col], index=False)
            hashed_features = pd.get_dummies(hashed_features % 100, prefix=f'{col}_hash')
            encoded_df = pd.concat([encoded_df, hashed_features], axis=1)

        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        logger.info("Categorical data handled")
        return df, None, f"Encoded {len(low_cardinality)} low cardinality and {len(high_cardinality)} high cardinality categorical columns."
    except Exception as e:
        logger.error(f"Error in handle_categorical_data: {str(e)}")
        logger.error(traceback.format_exc())
        return df, f"Error handling categorical data: {str(e)}", None

def auto_detect_and_convert_dtypes(df):
    for column in df.columns:
        try:
            if df[column].dtype == 'object':
                df[column] = pd.to_datetime(df[column], errors='ignore')
                if df[column].dtype != 'datetime64[ns]':
                    df[column] = pd.to_numeric(df[column], errors='ignore')
        except Exception as e:
            logger.error(f"Error converting column {column}: {e}")
    return df



def require_data(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = get_or_create_session_id()
        if not data_manager.is_data_available(session_id):
            return jsonify({'error': 'No data available. Please upload data first.'}), 400
        return f(*args, **kwargs)
    return decorated_function

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                logging.info(f"File saved to {file_path}")

                df = read_file(file_path)
                if df is None or df.empty:
                    raise ValueError("Failed to read data from the file or the file is empty")
                
                df = auto_detect_and_convert_dtypes(df)
                session_id = get_or_create_session_id()
                data_manager.set_data(session_id, df)
                session['has_data'] = True
                flash('File uploaded and data loaded successfully', 'success')
                return redirect(url_for('view'))
            except Exception as e:
                error_message = f"An error occurred while processing the file: {str(e)}"
                flash(error_message, 'error')
                logging.error(error_message, exc_info=True)
                return render_template('index.html', error=error_message)
    return render_template('index.html')





@app.route('/view')
@require_data
def view():
    try:
        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)
        if df is None or df.empty:
            return jsonify({"error": "No data available. Please upload data first."}), 400

        initial_data = prepare_data_for_json(df.head(50).to_dict(orient='records'))
        columns = df.columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        total_rows = len(df)

        return render_template('view.html',
                               columns=columns,
                               initial_data=json.dumps(initial_data),
                               categorical_columns=categorical_columns,
                               datetime_columns=datetime_columns,
                               numeric_columns=numeric_columns,
                               total_rows=total_rows)

    except Exception as e:
        app.logger.error(f"Error in view function: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    



@app.route('/download_cleaned_data')
@require_data
def download_cleaned_data():
    session_id = get_or_create_session_id()
    df = data_manager.get_data(session_id)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return send_file(temp_file.name,
                     mimetype='text/csv',
                     download_name='cleaned_data.csv',
                     as_attachment=True)

@app.route('/download_processed/<filename>')
def download_processed_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)




@app.route('/check_data_availability', methods=['GET'])
def check_data_availability():
    try:
        session_id = get_or_create_session_id()
        if data_manager.is_data_available(session_id):
            df = data_manager.get_data(session_id)
            if df is not None and not df.empty:
                return jsonify({
                    'status': 'available', 
                    'message': 'Data is available for transformation.',
                    'shape': df.shape,
                    'columns': df.columns.tolist()
                })
        
        return jsonify({
            'status': 'unavailable', 
            'message': 'No data available. Please upload data first.'
        }), 404
    except Exception as e:
        app.logger.error(f"Error in check_data_availability: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"An unexpected error occurred: {str(e)}"
        }), 500
    

# Socket.IO event handlers
@socketio.on('request_data')
def handle_request_data(message):
    try:
        start = int(message.get('start', 0))
        size = int(message.get('size', 50))
        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)
        if df is None:
            emit('data_update_error', {'error': 'No data available'})
            return
        data = prepare_data_for_json(df.iloc[start:start+size].to_dict(orient='records'))
        emit('data_update', {'data': data, 'last': start+size >= len(df)})
    except Exception as e:
        logging.error(f"Error handling data request: {str(e)}", exc_info=True)
        emit('data_update_error', {'error': str(e)})




@socketio.on('check_data_availability')
def handle_check_data_availability():
    session_id = get_or_create_session_id()
    df = data_manager.get_data(session_id)
   
    emit('data_availability', {'data_available': df is not None and not df.empty})

@socketio.on('apply_transformations')
def handle_apply_transformations(transformations):
    try:
        logging.info(f"Received transformations: {transformations}")
        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)
        
        if df is None or df.empty:
            raise ValueError("No data available to transform. Please upload data first.")
        
        emit_progress('Starting transformations...')
        logging.info(f"Initial DataFrame shape: {df.shape}")
        logging.info(f"Initial columns: {df.columns.tolist()}")

        pipeline = AdvancedDataPipeline(df)
        new_columns_created = []
        error_messages = []
        warning_messages = []

        # Extract similarity threshold for duplicate detection
        similarity_threshold = int(transformations.pop('similarity_threshold', 75))

        # Handle user-selected transformations
        for transform_type, columns in transformations.items():
            emit_progress(f'Applying {transform_type} transformation...')
            logging.info(f"Applying {transform_type} transformation to columns: {columns}")
            
            for column in columns:
                if column not in df.columns:
                    warning_messages.append(f"Column '{column}' not found. Skipping.")
                    continue
                
                try:
                    if transform_type == 'datetime_columns':
                        df, new_cols, _, warning = handle_datetime_column(df, column)
                    elif transform_type == 'categorical_columns':
                        df, new_cols, _, warning = encode_categorical_columns(df, [column])
                    elif transform_type == 'currency_columns':
                        df, new_cols, _, warning = handle_currency_column(df, column)
                    elif transform_type == 'address_columns':
                        df, new_cols, _, warning = handle_address_column(df, column)
                    else:
                        warning_messages.append(f"Unknown transformation type: {transform_type}. Skipping.")
                        continue

                    if warning:
                        warning_messages.append(warning)
                    new_columns_created.extend(new_cols)

                except Exception as e:
                    error_message = f"Error applying {transform_type} transformation to column '{column}': {str(e)}"
                    logging.error(error_message, exc_info=True)
                    error_messages.append(error_message)

        # Apply general data cleaning functions
        try:
            emit_progress('Handling inconsistent formats...')
            df, error, message = pipeline.handle_inconsistent_formats()
            if error:
                error_messages.append(error)
            elif message:
                logging.info(message)
            
            emit_progress('Handling missing values...')
            df, error, message = pipeline.handle_missing_values()
            if error:
                error_messages.append(error)
            elif message:
                logging.info(message)
            
            emit_progress('Handling outliers...')
            df, error, message = pipeline.handle_outliers()
            if error:
                error_messages.append(error)
            elif message:
                logging.info(message)
            
            emit_progress('Handling duplicates...')
            df, error, message = pipeline.handle_duplicates(similarity_threshold=similarity_threshold)
            if error:
                error_messages.append(error)
            elif message:
                logging.info(message)

        except Exception as e:
            error_message = f"Error during automatic pipeline transformations: {str(e)}"
            logging.error(error_message, exc_info=True)
            error_messages.append(error_message)

        # Update the pipeline and data manager with the transformed data
        pipeline.update_data(df)
        data_manager.update_data(session_id, df)  # Correct usage of update_data

        logging.info(f"Final DataFrame shape: {df.shape}")
        logging.info(f"Final columns: {df.columns.tolist()}")
        logging.info(f"New columns created: {new_columns_created}")
        data_manager.set_transformed_data(session_id, df.copy())

        response_data = prepare_data_for_json(df.head(50).to_dict(orient='records'))

        if error_messages:
            socketio.emit('transformation_error', {'errors': error_messages})

        if warning_messages:
            socketio.emit('transformation_warning', {'warnings': warning_messages})

        socketio.emit('transformation_complete', {
            'columns': df.columns.tolist(),
            'data': response_data,
            'new_columns': new_columns_created,
            'total_rows': len(df)
        })

    except ValueError as ve:
        error_message = str(ve)
        logging.error(f"ValueError in handle_apply_transformations: {error_message}")
        socketio.emit('transformation_error', {'error': error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(f"Unexpected error in handle_apply_transformations: {str(e)}", exc_info=True)
        socketio.emit('transformation_error', {'error': error_message})







@app.route('/reset_data_manager', methods=['POST'])
def reset_data_manager():
    try:
        data_manager.clear_data()
        return jsonify({'message': 'Data manager reset successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error resetting data manager: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500






def get_or_create_session_id(provided_session_id=None):
    if provided_session_id:
        return provided_session_id
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']



        




@socketio.on('apply_advanced_transformations')
def handle_apply_advanced_transformations(data):
    try:
        session_id = get_or_create_session_id()
        transformed_df = data_manager.get_transformed_data(session_id)
        
        if transformed_df is None or transformed_df.empty:
            raise ValueError("No transformed data available. Please apply initial transformations first.")
        
        advanced_operations = {
            'high_dimensionality': AdvancedDataPipeline(transformed_df.copy()).handle_high_dimensionality,
            'scale_transform': AdvancedDataPipeline(transformed_df.copy()).scale_and_transform_data,
            'feature_engineering': AdvancedDataPipeline(transformed_df.copy()).feature_engineering,
            'dynamic_feature': AdvancedDataPipeline(transformed_df.copy()).dynamic_feature_transformation,
            'advanced_transform': AdvancedDataPipeline(transformed_df.copy()).advanced_data_transformation
        }
        
        for transform in data['transforms']:
            if transform in advanced_operations:
                emit_progress(f'Performing {transform}...')
                try:
                    result_df, error, message = advanced_operations[transform]()
                    
                    if error:
                        logging.error(f"Error in {transform}: {error}")
                        socketio.emit('advanced_transformation_error', {'transform': transform, 'error': error})
                    elif message:
                        logging.info(f"Message from {transform}: {message}")
                    
                    if result_df is not None and not result_df.empty:
                        result_data = prepare_data_for_json(result_df.head(50).to_dict(orient='records'))
                        
                        # Save the result to a CSV file
                        filename = f"{transform}_result.csv"
                        result_df.to_csv(os.path.join(app.config['PROCESSED_FOLDER'], filename), index=False)
                        
                        socketio.emit('advanced_transformation_result', {
                            'transform': transform,
                            'result': {
                                'columns': result_df.columns.tolist(),
                                'data': result_data,
                                'total_rows': len(result_df),
                                'download_url': url_for('download_processed_file', filename=filename)
                            }
                        })
                    else:
                        socketio.emit('advanced_transformation_result', {
                            'transform': transform,
                            'result': {
                                'message': 'No changes were made to the data.'
                            }
                        })
                except Exception as e:
                    error_message = f"Error in {transform}: {str(e)}"
                    logging.error(error_message, exc_info=True)
                    socketio.emit('advanced_transformation_error', {'transform': transform, 'error': error_message})
            else:
                socketio.emit('advanced_transformation_error', {
                    'transform': transform,
                    'error': f'Unknown transformation: {transform}'
                })
        
        socketio.emit('all_advanced_transformations_complete')
    
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(f"Unexpected error in handle_apply_advanced_transformations: {str(e)}", exc_info=True)
        socketio.emit('transformation_error', {'error': error_message})



@socketio.on('generate_insights')
def handle_generate_insights():
    try:
        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)

        
        if df is None or df.empty:
            raise ValueError("No data available to generate insights. Please upload data first.")
        
        emit_progress('Generating insights...')
        logging.info("Generating insights")

        pipeline = AdvancedDataPipeline(df)
        insights = pipeline.generate_insights()
        
        emit('insights_update', {
            'insights': insights
        })
        emit_progress('Insights generation complete!')

    except ValueError as ve:
        logging.error(f"ValueError in handle_generate_insights: {str(ve)}")
        emit('insights_error', {'error': str(ve)})
    except Exception as e:
        logging.error(f"Unexpected error in handle_generate_insights: {str(e)}", exc_info=True)
        emit('insights_error', {'error': f"An unexpected error occurred: {str(e)}"})

@socketio.on('generate_visualizations')
def handle_generate_visualizations():
    try:
        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)

        
        if df is None or df.empty:
            raise ValueError("No data available to generate visualizations. Please upload data first.")
        
        emit_progress('Generating visualizations...')
        logging.info("Starting visualization generation")

        pipeline = AdvancedDataPipeline(df)
        visualizations, attempted = pipeline.generate_visualizations()
        
        logging.info(f"Attempted to generate {len(attempted)} visualizations: {', '.join(attempted)}")
        logging.info(f"Successfully generated {len(visualizations)} visualizations")

        if visualizations:
            viz_json = []
            for viz in visualizations:
                viz_name, viz_fig = viz
                if viz_fig is not None:
                    try:
                        json_data = viz_fig.to_json()
                        viz_json.append((viz_name, json_data))
                        logging.info(f"Visualization type: {viz_name}, JSON created (length: {len(json_data)})")
                    except Exception as json_error:
                        logging.error(f"Error converting {viz_name} to JSON: {str(json_error)}")
                else:
                    logging.warning(f"Visualization type: {viz_name} is None, skipping")
            
            logging.info(f"Prepared {len(viz_json)} visualizations for emission")
            emit('visualizations_update', {
                'visualizations': viz_json,
                'message': f"Generated {len(viz_json)} visualizations out of {len(attempted)} attempted"
            })
            logging.info("Emitted visualizations_update event")
        else:
            logging.warning("No visualizations generated")
            emit('visualizations_error', {'error': f"No visualizations could be generated. Attempted: {', '.join(attempted)}"})

        emit_progress('Visualizations generation complete!')

    except Exception as e:
        logging.error(f"Unexpected error in handle_generate_visualizations: {str(e)}", exc_info=True)
        emit('visualizations_error', {'error': f"An unexpected error occurred: {str(e)}"})










@socketio.on('train_ml_model')
def handle_train_ml_model(data):
    try:
        logging.info("Starting ML model training")
        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)
        if df is None or df.empty:
            raise ValueError("No data available to train model. Please upload data first.")
        
        target_column = data.get('target_column')
        if not target_column:
            raise ValueError("No target column specified for training.")
        
        emit_progress('Training ML model...')
        pipeline = AdvancedDataPipeline(df)
        pipeline.universal_aiml.set_session_id(session_id)
        model_results = pipeline.build_ml_model(target_column)
        
        logging.info(f"Model training completed. Results: {model_results}")
        
        evaluation_results = pipeline.universal_aiml.evaluate_model()
        logging.info(f"Model evaluation results: {evaluation_results}")

        # Store the trained model in the session
        session['trained_model'] = pipeline.universal_aiml

        serialized_results = safe_serialize(evaluation_results)
        serialized_training_result = safe_serialize(model_results)
        
        emit('ml_model_results', {'results': serialized_results, 'training_result': serialized_training_result})
        emit('model_trained', {'features': safe_serialize(pipeline.universal_aiml.feature_columns)})
        emit_progress('ML model training complete!')

    except Exception as e:
        logging.error(f"Error in handle_train_ml_model: {str(e)}", exc_info=True)
        emit('ml_model_error', {'error': f"An error occurred: {str(e)}"})




@socketio.on('make_predictions')
def handle_make_predictions(data):
    try:
        session_id = get_or_create_session_id()
        universal_aiml = session.get('trained_model')
        if universal_aiml is None:
            raise ValueError("No trained model available. Please train a model first.")

        start_date = pd.to_datetime(data['startDate'])
        end_date = pd.to_datetime(data['endDate'])
        date_range = pd.date_range(start=start_date, end=end_date)

        if universal_aiml.is_time_series:
            predictions = universal_aiml.make_predictions(date_range)
        else:
            feature_data = generate_feature_data(date_range, universal_aiml.feature_columns, universal_aiml.df)
            predictions = universal_aiml.make_predictions(feature_data)

        # Safely serialize the predictions
        serialized_predictions = safe_serialize(predictions)

        emit('prediction_results', {'predictions': serialized_predictions})
    except Exception as e:
        logging.error(f"Error in handle_make_predictions: {str(e)}", exc_info=True)
        emit('prediction_error', {'error': f"An error occurred: {str(e)}"})




def generate_feature_data(date_range, feature_columns, original_df):
    # Generate feature data based on the original dataset's patterns
    feature_data = pd.DataFrame(index=date_range)
    for column in feature_columns:
        if column in original_df.select_dtypes(include=['datetime64']).columns:
            feature_data[column] = date_range
        elif column in original_df.select_dtypes(include=[np.number]).columns:
            # Use a simple moving average for numeric columns
            values = original_df[column].rolling(window=7).mean().dropna().tolist()
            feature_data[column] = np.random.choice(values, size=len(date_range))
        else:
            # For categorical columns, sample from the original distribution
            feature_data[column] = np.random.choice(original_df[column].dropna().unique(), size=len(date_range))
    return feature_data

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 16 MB.', 'error')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('index')), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify(error=str(e)), 500







class ResponseFormatter:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.importance_threshold = 0.5

    def extract_key_phrases(self, text: str) -> List[str]:
        # Extract key phrases using TF-IDF
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        important_phrase_indices = tfidf_matrix.sum(axis=0).A1 > self.importance_threshold
        return [feature_names[i] for i in range(len(feature_names)) if important_phrase_indices[i]]

    def format_response(self, response: str) -> str:
        try:
            sentences = sent_tokenize(response)
        except LookupError:
            logger.warning("NLTK punkt tokenizer not found. Falling back to basic sentence splitting.")
            sentences = response.split('. ')
        
        key_phrases = self.extract_key_phrases(response)
        
        formatted_lines = []
        current_section = ""
        
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in ["analysis", "summary", "conclusion"]):
                if current_section:
                    formatted_lines.append("</ul>")
                formatted_lines.append(f"<h3>{sentence}</h3>")
                formatted_lines.append("<ul>")
                current_section = sentence
            else:
                formatted_sentence = sentence
                for phrase in key_phrases:
                    if phrase.lower() in formatted_sentence.lower():
                        formatted_sentence = re.sub(
                            f"({re.escape(phrase)})",
                            r"<strong>\1</strong>",
                            formatted_sentence,
                            flags=re.IGNORECASE
                        )
                formatted_lines.append(f"<li>{formatted_sentence}</li>")
        
        if current_section:
            formatted_lines.append("</ul>")
        
        return "\n".join(formatted_lines)


class VisualizationGenerator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def generate_visualizations(self, user_input: str) -> List[Tuple[str, Any]]:
        try:
            return self.pipeline.generate_chat_visualizations(user_input)
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
            return []

class ChatHandler:
    def __init__(self, data_manager, response_formatter: ResponseFormatter, viz_generator: VisualizationGenerator):
        self.data_manager = data_manager
        self.response_formatter = response_formatter
        self.viz_generator = viz_generator
        self.executor = ThreadPoolExecutor(max_workers=3)

    def handle_chat_message(self, message: Dict[str, Any]):
        try:
            df = self.data_manager.get_data()
            if df is None or df.empty:
                raise ValueError("No data available. Please upload data first.")

            user_input = message['text']
            sentiment = TextBlob(user_input).sentiment.polarity

            pipeline = AdvancedDataPipeline(df)
            response = pipeline.chat_with_data(user_input)
            formatted_response = self.response_formatter.format_response(response)

            self.emit_response(formatted_response, sentiment)

            visualizations = pipeline.generate_chat_visualizations(user_input)
            self.emit_visualizations(visualizations)

        except Exception as e:
            logger.error(f"Error in handle_chat_message: {str(e)}", exc_info=True)
            error_message = self.generate_error_message(str(e))
            self.emit_error(error_message)


    def emit_response(self, response: str, sentiment: float):
        chunks = self.chunk_response(response)
        for chunk in chunks:
            socketio.emit('chat_response_chunk', {
                'chunk': chunk,
                'sentiment': sentiment
            })
            time.sleep(0.1)  # Adjust for desired speed
        socketio.emit('chat_response_end')

    def emit_visualizations(self, visualizations: List[Tuple[str, Any]]):
        for viz_type, viz_data in visualizations:
            socketio.emit('chat_visualization', {
                'type': viz_type,
                'data': viz_data
            })

    def emit_error(self, error_message: str):
        socketio.emit('chat_response_chunk', {'chunk': error_message})
        socketio.emit('chat_response_end')


    def chunk_response(self, response: str, chunk_size: int = 100) -> List[str]:
        return [response[i:i+chunk_size] for i in range(0, len(response), chunk_size)]

    def generate_error_message(self, error: str) -> str:
        generic_message = "I apologize, but I'm having trouble processing your request."
        if "no data available" in error.lower():
            return f"{generic_message} It seems there's no data uploaded yet. Please upload some data and try again."
        elif "connection" in error.lower():
            return f"{generic_message} There might be a connection issue. Please check your internet connection and try again."
        else:
            return f"{generic_message} Please try again later or rephrase your question."



# Socket.IO event handler
@socketio.on('chat_message')
def handle_chat_message(message):
    try:
        session_id = get_or_create_session_id()
        df = data_manager.get_transformed_data(session_id)
        if df is None or df.empty:
            raise ValueError("No data available. Please upload and transform data first.")
        
        pipeline = AdvancedDataPipeline(df)
        user_input = message['text']
        
        response = pipeline.chat_with_data(user_input)
        emit('chat_response_chunk', {'chunk': response})
        
        visualizations = pipeline.generate_chat_visualizations(user_input)
        for viz in visualizations:
            viz_type, viz_data = viz
            emit('chat_visualization', {
                'type': viz_type,
                'data': viz_data
            })
        
        emit('chat_response_end')
    
    except Exception as e:
        logging.error(f"Error in handle_chat_message: {str(e)}", exc_info=True)
        emit('chat_response_chunk', {'chunk': "<p>I apologize, but I'm having trouble processing your request at the moment. Could you please try again later?</p>"})
        emit('chat_response_end')







report_progress_dict = {}
report_pdfs = {}
report_errors = {}









class AdvancedReportGeneration:
    def __init__(self, data_manager, model_results=None):
        self.data_manager = data_manager
        self.model_results = model_results
        self.logger = logging.getLogger(__name__)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.pdf_buffer = io.BytesIO()
        self.doc = SimpleDocTemplate(self.pdf_buffer, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.elements = []

    def add_data_summary(self):
        df = self.data_manager.get_data(self.data_manager.get_current_session_id())
        summary = df.describe().to_dict()
        self.elements.append(Paragraph("Data Summary", self.styles['Heading1']))
        for column, stats in summary.items():
            self.elements.append(Paragraph(f"{column}:", self.styles['Heading2']))
            for stat, value in stats.items():
                self.elements.append(Paragraph(f"{stat}: {value:.2f}", self.styles['Normal']))
            self.elements.append(Spacer(1, 12))

    def generate_ai_insights(self, df):
        try:
            data_summary = df.describe().to_string()
            column_info = "\n".join([f"{col}: {df[col].dtype}" for col in df.columns])
            
            prompt = f"""Analyze the following dataset summary and provide key insights:

            Dataset Summary:
            {data_summary}

            Column Information:
            {column_info}

            Please provide:
            1. Key insights about the data
            2. Potential patterns or trends
            3. Recommendations for further analysis
            4. Any potential issues or areas of concern
            5. Suggestions for feature engineering or model selection based on the available data
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating AI insights: {str(e)}")
            return "Error generating AI insights."

    def create_visualizations(self, df):
        visualizations = []
        try:
            # Histogram for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                visualizations.append(('histogram', col, self.fig_to_img(fig)))

            # Correlation heatmap for numeric columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                ax.set_title('Correlation Heatmap')
                visualizations.append(('heatmap', 'Correlation Heatmap', self.fig_to_img(fig)))

            # Bar chart for categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
            for col in cat_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                visualizations.append(('bar', col, self.fig_to_img(fig)))

            # Scatter plot for pairs of numeric columns
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=ax)
                ax.set_title(f'Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}')
                visualizations.append(('scatter', f'{numeric_cols[0]} vs {numeric_cols[1]}', self.fig_to_img(fig)))

            # Line chart for numeric columns (assuming the first column is a time series)
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(12, 6))
                for col in numeric_cols[1:3]:  # Plot up to 2 numeric columns
                    df.plot(x=numeric_cols[0], y=col, ax=ax, label=col)
                ax.set_title(f'Line Chart: {numeric_cols[0]} vs Other Numeric Columns')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel('Value')
                ax.legend()
                visualizations.append(('line', f'{numeric_cols[0]} vs Other Columns', self.fig_to_img(fig)))

            # Pair plot for numeric columns
            if len(numeric_cols) >= 2:
                fig = sns.pairplot(df[numeric_cols[:4]], diag_kind='kde')  # Limit to 4 columns for readability
                fig.fig.suptitle('Pair Plot of Numeric Variables', y=1.02)
                visualizations.append(('pairplot', 'Numeric Variables', self.fig_to_img(fig.fig)))

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")

        return visualizations




    

    def fig_to_img(self, fig):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer



    def perform_custom_analysis(self, df, custom_requests):
        results = []
        if not custom_requests:
            return results

        for request in custom_requests.split('\n'):
            try:
                self.logger.info(f"Processing custom analysis request: {request}")
                result, fig = self.analyze_request(df, request)
                img_buffer = self.fig_to_img(fig)
                results.append((request, result, img_buffer))
            except Exception as e:
                self.logger.error(f"Error in custom analysis: {str(e)}")
                self.logger.error(traceback.format_exc())
                results.append((request, f"Error: {str(e)}", None))

        return results

    def analyze_request(self, df, request):
        words = request.lower().split()
        columns = [col for col in df.columns if col.lower() in words]

        if not columns:
            return "No valid column names found in the request. Please check the column names and try again.", self.create_error_figure("No valid columns")

        if len(columns) == 1:
            return self.single_column_analysis(df, columns[0])
        elif len(columns) == 2:
            return self.two_column_analysis(df, columns[0], columns[1])
        else:
            return self.multi_column_analysis(df, columns)

    def single_column_analysis(self, df, column):
        if column not in df.columns:
            return f"Column '{column}' not found in the dataset.", self.create_error_figure(f"Column '{column}' not found")

        result = f"Analysis for column: {column}\n\n"
        fig, ax = plt.subplots(figsize=(10, 6))

        if df[column].dtype in ['int64', 'float64']:
            result += f"Numeric column analysis:\n"
            result += f"Mean: {df[column].mean():.2f}\n"
            result += f"Median: {df[column].median():.2f}\n"
            result += f"Standard Deviation: {df[column].std():.2f}\n"
            result += f"Min: {df[column].min():.2f}\n"
            result += f"Max: {df[column].max():.2f}\n"

            sns.histplot(df[column], kde=True, ax=ax)
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')

        elif df[column].dtype == 'object' or df[column].dtype.name == 'category':
            result += f"Categorical column analysis:\n"
            value_counts = df[column].value_counts()
            result += f"Unique values: {len(value_counts)}\n"
            result += f"Top 5 categories:\n{value_counts.head().to_string()}\n"

            sns.countplot(y=column, data=df, order=value_counts.index[:10], ax=ax)
            ax.set_title(f'Top 10 Categories in {column}')
            ax.set_xlabel('Count')
            ax.set_ylabel(column)

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            result += f"Datetime column analysis:\n"
            result += f"Date range: {df[column].min()} to {df[column].max()}\n"
            
            df[column].value_counts().resample('Y').sum().plot(kind='line', ax=ax)
            ax.set_title(f'Yearly Trend of {column}')
            ax.set_xlabel('Year')
            ax.set_ylabel('Count')

        else:
            return f"Unable to analyze column '{column}'. Unsupported data type.", self.create_error_figure(f"Unsupported data type for '{column}'")

        return result, fig

    def two_column_analysis(self, df, col1, col2):
        if col1 not in df.columns or col2 not in df.columns:
            return f"One or both columns not found in the dataset.", self.create_error_figure("Column(s) not found")

        result = f"Analysis for columns: {col1} and {col2}\n\n"
        fig, ax = plt.subplots(figsize=(10, 6))

        if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
            result += f"Correlation coefficient: {df[col1].corr(df[col2]):.2f}\n"
            sns.scatterplot(x=col1, y=col2, data=df, ax=ax)
            ax.set_title(f'Scatter plot: {col1} vs {col2}')

        elif (df[col1].dtype in ['int64', 'float64'] and (df[col2].dtype == 'object' or df[col2].dtype.name == 'category')) or \
             (df[col2].dtype in ['int64', 'float64'] and (df[col1].dtype == 'object' or df[col1].dtype.name == 'category')):
            num_col, cat_col = (col1, col2) if df[col1].dtype in ['int64', 'float64'] else (col2, col1)
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
            ax.set_title(f'Box plot: {num_col} by {cat_col}')
            result += f"Analysis of {num_col} grouped by {cat_col}:\n"
            result += df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).to_string()

        elif (df[col1].dtype == 'object' or df[col1].dtype.name == 'category') and (df[col2].dtype == 'object' or df[col2].dtype.name == 'category'):
            contingency_table = pd.crosstab(df[col1], df[col2])
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            ax.set_title(f'Heatmap: {col1} vs {col2}')
            result += f"Contingency table:\n{contingency_table.to_string()}\n"
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            result += f"Chi-square test p-value: {p_value:.4f}\n"

        else:
            return f"Unable to analyze columns '{col1}' and '{col2}' together. Unsupported combination of data types.", self.create_error_figure("Unsupported data types")

        return result, fig

    def multi_column_analysis(self, df, columns):
        result = f"Multi-column analysis for: {', '.join(columns)}\n\n"
        
        numeric_columns = [col for col in columns if df[col].dtype in ['int64', 'float64']]
        categorical_columns = [col for col in columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']

        if len(numeric_columns) >= 2:
            corr_matrix = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            result += f"Correlation matrix:\n{corr_matrix.to_string()}\n\n"
        elif len(categorical_columns) >= 2:
            fig, ax = plt.subplots(figsize=(12, 6))
            df[categorical_columns].nunique().plot(kind='bar', ax=ax)
            ax.set_title('Number of Unique Values in Categorical Columns')
            ax.set_ylabel('Number of Unique Values')
            result += "Unique value counts for categorical columns:\n"
            result += df[categorical_columns].nunique().to_string() + "\n\n"
        else:
            return f"Unable to perform multi-column analysis. Please provide at least two numeric or two categorical columns.", self.create_error_figure("Insufficient columns for analysis")

        return result, fig

    def create_error_figure(self, error_message):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, error_message, ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig










    def perform_statistical_analysis(self, df):
        analysis_results = {}
        try:
            # Descriptive statistics
            analysis_results['descriptive'] = df.describe().to_dict()

            # Feature importance (for numeric columns)
            target = df.columns[-1]  # Assuming last column is the target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors='ignore')
            if len(numeric_cols) > 0:
                mi_scores = mutual_info_regression(df[numeric_cols], df[target])
                analysis_results['feature_importance'] = dict(zip(numeric_cols, mi_scores))

            # Correlation analysis
            analysis_results['correlation'] = df.corr().to_dict()

            # Skewness and Kurtosis
            analysis_results['skewness'] = df.skew().to_dict()
            analysis_results['kurtosis'] = df.kurtosis().to_dict()

        except Exception as e:
            self.logger.error(f"Error performing statistical analysis: {str(e)}")

        return analysis_results

    def perform_time_series_analysis(self, df):
        time_series_results = {}
        try:
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0:
                date_col = date_columns[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                    df_ts = df[[date_col, target_col]].set_index(date_col)
                    df_ts = df_ts.resample('D').mean()  # Resample to daily frequency
                    result = seasonal_decompose(df_ts, model='additive')
                    
                    plt.figure(figsize=(12, 10))
                    result.plot()
                    plt.tight_layout()
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png')
                    img_buffer.seek(0)
                    time_series_results['decomposition'] = img_buffer
                    plt.close()

        except Exception as e:
            self.logger.error(f"Error performing time series analysis: {str(e)}")

        return time_series_results

    def perform_sentiment_analysis(self, df):
        sentiment_results = {}
        try:
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                for col in text_columns:
                    sentiments = df[col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                    sentiment_results[col] = {
                        'mean': sentiments.mean(),
                        'median': sentiments.median(),
                        'min': sentiments.min(),
                        'max': sentiments.max()
                    }
                    
                    plt.figure(figsize=(10, 6))
                    sns.histplot(sentiments, kde=True)
                    plt.title(f'Sentiment Distribution for {col}')
                    plt.xlabel('Sentiment Polarity')
                    plt.ylabel('Frequency')
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png')
                    img_buffer.seek(0)
                    sentiment_results[f'{col}_distribution'] = img_buffer
                    plt.close()

        except Exception as e:
            self.logger.error(f"Error performing sentiment analysis: {str(e)}")

        return sentiment_results

    def add_page_number(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 10)
        page_number_text = f"Page {doc.page}"
        canvas.drawCentredString(4*inch, 0.25*inch, page_number_text)
        canvas.restoreState()

    def compile_report(self, ai_insights, visualizations, custom_analysis_results):
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create a custom style for captions
            styles.add(ParagraphStyle(name='Caption',
                                      parent=styles['Normal'],
                                      fontSize=10,
                                      leading=12,
                                      alignment=TA_CENTER,
                                      spaceAfter=6))

            elements = []

            # Title
            elements.append(Paragraph("Advanced Data Analysis Report", styles['Title']))
            elements.append(Spacer(1, 12))

            # AI Insights
            elements.append(Paragraph("AI-Generated Insights", styles['Heading1']))
            elements.append(Paragraph(ai_insights, styles['BodyText']))
            elements.append(Spacer(1, 12))

            # Visualizations
            elements.append(Paragraph("Data Visualizations", styles['Heading1']))
            for viz_type, viz_name, img_buffer in visualizations:
                img = Image(img_buffer, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Paragraph(f"{viz_type.capitalize()}: {viz_name}", styles['Caption']))
                elements.append(Spacer(1, 12))

            # Custom Analysis Results
            if custom_analysis_results:
                elements.append(Paragraph("Custom Analysis", styles['Heading1']))
                for request, result, img_buffer in custom_analysis_results:
                    elements.append(Paragraph(f"Request: {request}", styles['Heading3']))
                    elements.append(Paragraph(result, styles['BodyText']))
                    if img_buffer:
                        img = Image(img_buffer, width=6*inch, height=4*inch)
                        elements.append(img)
                    elements.append(Spacer(1, 12))

            doc.build(elements)
            pdf_content = buffer.getvalue()
            buffer.close()

            return pdf_content

        except Exception as e:
            self.logger.error(f"Error compiling report: {str(e)}")
            raise

    



    def generate_final_report(self, df, ai_insights, visualizations, custom_analysis_results):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
        elements = []

        # Title
        elements.append(Paragraph("Advanced Data Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Dataset Overview
        elements.append(Paragraph("Dataset Overview", styles['Heading1']))
        elements.append(Paragraph(f"Number of Rows: {df.shape[0]}", styles['BodyText']))
        elements.append(Paragraph(f"Number of Columns: {df.shape[1]}", styles['BodyText']))
        elements.append(Paragraph("Columns:", styles['BodyText']))
        for col in df.columns:
            elements.append(Paragraph(f"- {col} ({df[col].dtype})", styles['BodyText']))
        elements.append(Spacer(1, 12))

        # AI-Generated Insights
        elements.append(Paragraph("AI-Generated Insights", styles['Heading1']))
        elements.append(Paragraph(ai_insights, styles['BodyText']))
        elements.append(Spacer(1, 12))

        # Data Visualizations
        elements.append(Paragraph("Data Visualizations", styles['Heading1']))
        for viz_type, viz_name, img_buffer in visualizations:
            elements.append(Paragraph(f"{viz_type.capitalize()}: {viz_name}", styles['Heading2']))
            img = Image(img_buffer, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 12))

        elements.append(PageBreak())

        # Custom Analysis Results
        elements.append(Paragraph("Custom Analysis Results", styles['Heading1']))
        for request, result, img_buffer in custom_analysis_results:
            elements.append(Paragraph(f"Analysis Request: {request}", styles['Heading2']))
            elements.append(Spacer(1, 6))
            
            # Add the textual result
            result_paragraphs = result.split('\n')
            for para in result_paragraphs:
                elements.append(Paragraph(para, styles['BodyText']))
                elements.append(Spacer(1, 3))
            
            # Add the visualization if available
            if img_buffer:
                img = Image(img_buffer, width=6*inch, height=4*inch)
                elements.append(img)
            
            elements.append(Spacer(1, 12))

        # Build the PDF
        doc.build(elements)
        pdf_content = buffer.getvalue()
        buffer.close()

        return pdf_content








    def generate_report(self, custom_requests=None, selected_columns=None, session_id=None):
        try:
            self.logger.info("Starting report generation")
            self.logger.info(f"Custom requests: {custom_requests}")

            df = self.data_manager.get_data(session_id)
            
            if df is None or df.empty:
                raise ValueError("No data available for report generation")

            self.logger.info(f"Available columns in DataFrame: {df.columns.tolist()}")

            # Generate AI insights based on the available data
            ai_insights = self.generate_ai_insights(df)

            # Create visualizations
            visualizations = self.create_visualizations(df)

            # Perform custom analysis
            custom_analysis_results = self.perform_custom_analysis(df, custom_requests)

            # Compile the report
            pdf_content = self.compile_report(ai_insights, visualizations, custom_analysis_results)

            return pdf_content

        except Exception as e:
            self.logger.error(f"Error in generate_report: {str(e)}", exc_info=True)
            raise


def get_or_create_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_report_progress(report_id):
    return report_progress_dict.get(report_id, 0)

def update_report_progress(report_id, progress):
    report_progress_dict[report_id] = progress
    socketio.emit('report_progress', {'report_id': report_id, 'progress': progress}, namespace='/report')



def generate_report_task(report_id, data_manager, custom_requests, selected_columns, session_id):
    try:
        update_report_progress(report_id, 10)
        
        df = data_manager.get_data(session_id)
        report_generator = AdvancedReportGeneration(data_manager)
        update_report_progress(report_id, 20)
        
        logger.info(f"Starting report generation for report_id: {report_id}")
        logger.info(f"Custom requests: {custom_requests}")
        logger.info(f"Selected columns: {selected_columns}")
        
        # Generate AI insights
        ai_insights = report_generator.generate_ai_insights(df)
        update_report_progress(report_id, 40)
        
        # Generate general visualizations
        visualizations = report_generator.create_visualizations(df)
        update_report_progress(report_id, 60)
        
        # Perform custom analysis
        custom_analysis_results = report_generator.perform_custom_analysis(df, custom_requests)
        update_report_progress(report_id, 80)
        
        # Generate final report
        pdf_content = report_generator.generate_final_report(df, ai_insights, visualizations, custom_analysis_results)
        
        if pdf_content:
            report_pdfs[report_id] = pdf_content
            update_report_progress(report_id, 100)
            logger.info(f"Report generated successfully for report_id: {report_id}")
        else:
            raise Exception("Failed to generate PDF content")
    except Exception as e:
        logger.error(f"Error generating report for report_id {report_id}: {str(e)}")
        update_report_progress(report_id, -1)  # Indicate error
        report_errors[report_id] = str(e)  # Store the error message





@app.route('/api/report_error/<report_id>', methods=['GET'])
def api_report_error(report_id):
    error_message = report_errors.get(report_id, "Unknown error occurred")
    return jsonify({'error': error_message})


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        custom_requests = data.get('custom_requests', None)
        selected_columns = data.get('selected_columns', None)

        session_id = get_or_create_session_id()
        df = data_manager.get_data(session_id)
        
        if df is None or df.empty:
            return jsonify({'error': 'No data available. Please upload data first.'}), 400

        report_generator = AdvancedReportGeneration(data_manager)
        pdf_content = report_generator.generate_report(custom_requests, selected_columns)

        if pdf_content:
            response = make_response(pdf_content)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = 'inline; filename=advanced_data_report.pdf'
            return response
        else:
            return jsonify({'error': 'Failed to generate report'}), 500

    except Exception as e:
        app.logger.error(f"Error in generate_report route: {str(e)}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/report_progress', methods=['GET'])
def report_progress():
    report_id = request.args.get('report_id')
    if not report_id:
        return jsonify({'error': 'No report_id provided'}), 400
    progress = get_report_progress(report_id)
    return jsonify({'progress': progress})



@socketio.on('generate_report')
def handle_generate_report(data):
    custom_requests = data.get('custom_requests', '')
    selected_columns = data.get('selected_columns', None)
    report_id = str(uuid.uuid4())
    
    socketio.start_background_task(generate_report_task, report_id, data_manager, custom_requests, selected_columns)
    
    emit('report_generation_started', {'report_id': report_id})




@app.route('/api/generate_report', methods=['POST'])
def api_generate_report():
    try:
        data = request.json
        custom_requests = data.get('custom_requests', '')
        selected_columns = data.get('selected_columns', None)

        session_id = get_or_create_session_id()
        report_id = str(uuid.uuid4())
        
        # Start the report generation in a separate thread
        thread = threading.Thread(target=generate_report_task, args=(report_id, data_manager, custom_requests, selected_columns, session_id))
        thread.start()

        return jsonify({'report_id': report_id, 'message': 'Report generation started'}), 202

    except Exception as e:
        logger.error(f"Error initiating report generation: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/report_progress/<report_id>', methods=['GET'])
def api_report_progress(report_id):
    progress = get_report_progress(report_id)
    return jsonify({'progress': progress})



@app.route('/api/download_report/<report_id>', methods=['GET'])
def api_download_report(report_id):
    if report_id in report_pdfs:
        pdf_content = report_pdfs[report_id]
        
        # Save the PDF content to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(pdf_content)
        temp_file.close()

        return send_file(temp_file.name, as_attachment=True, download_name='advanced_data_report.pdf')
    elif get_report_progress(report_id) == -1:
        return jsonify({'error': 'Report generation failed'}), 500
    else:
        return jsonify({'error': 'Report not ready yet'}), 404







@socketio.on('connect', namespace='/report')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect', namespace='/report')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

# Main execution
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)