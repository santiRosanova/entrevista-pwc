import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):

    df = df.copy()

    # One-hot encoding categorical variables
    categorical_features = ["Gender", "Education Level", "Job Title"]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_features])
    categorical_columns = encoder.get_feature_names_out(categorical_features)
    df_encoded = pd.DataFrame(encoded_categorical, columns=categorical_columns)
    
    # Scaling numerical variables
    numerical_features = ["Age", "Years of Experience"]
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_features])
    df_scaled = pd.DataFrame(scaled_numerical, columns=numerical_features)
    
    X = pd.concat([df_scaled, df_encoded], axis=1)
    y = df["Salary"].values
    
    return X, y, encoder, scaler

def split_data(X, y, test_size=0.2, random_state=1):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    # Chose to train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluating the model with MAE and RMSE, two common metrics for regression
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    return mae, rmse

def train_baseline(X_train, y_train):
    # Training a Dummy Regressor to compare with the model
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    return baseline

def evaluate_baseline(baseline, X_test, y_test):
    # Evaluating the baseline with MAE and RMSE
    predictions = baseline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    return mae, rmse
