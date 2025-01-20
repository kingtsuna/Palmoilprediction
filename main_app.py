import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import numpy as np
warnings.filterwarnings('ignore')


# Trading Strategy Functions
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Date_copy'] = df['Date']
    df.set_index('Date', inplace=True)
    df['Month'] = df.index.strftime('%b-%Y')
    df = df.sort_values(by="Date")
    return df

def evaluate_strategy_success(df, signal_column, success_days_min, success_days_max, success_threshold):
    """Evaluate strategy success based on custom price movement criteria"""
    success_count = 0
    total_signals = 0
    details = []
    print(df.columns)
    for i in range(len(df) - success_days_max):

        if df[signal_column].iloc[i] != 0:
            total_signals += 1
            entry_price = df['Price'].iloc[i]

            # Check prices from day 'success_days_min' to 'success_days_max'
            for j in range(i + success_days_min, i + success_days_max + 1):
                exit_price = df['Price'].iloc[j]
                
                exit_date = df['Date_copy'].iloc[j]
                price_change = ((exit_price - entry_price) / entry_price) * 100

                success = False
                if (df[signal_column].iloc[i] == -1 and price_change >= success_threshold) or \
                   (df[signal_column].iloc[i] == 1 and price_change <= -success_threshold):
                    success_count += 1
                    success = True
                    break  # Exit the inner loop as soon as a successful trade is found

            details.append({
                'Date': df.index[i],
                'signal_column': df[signal_column].iloc[i],
                'Signal': 'Buy' if df[signal_column].iloc[i] == -1 else 'Sell',
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Price_Change%': round(price_change, 2),
                'Success': success,
                'exit_date':exit_date
            })

    success_rate = (success_count / total_signals * 100) if total_signals > 0 else 0
    return success_rate, success_count, total_signals, details

def calculate_stochastic_signals(df):
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Price'])
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    df['stoch_signal'] = 0
    df.loc[(df['%K'] > df['%D']) & (df['%K'] < 30), 'stoch_signal'] = 1
    df.loc[(df['%K'] < df['%D']) & (df['%K'] > 70), 'stoch_signal'] = -1
    return df

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd_signals(df):
    df['ema_12'] = calculate_ema(df['Price'], span=12)
    df['ema_26'] = calculate_ema(df['Price'], span=26)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal_line'] = calculate_ema(df['macd'], span=9)
    df['macd_signal'] = 0
    df.loc[df['macd'] > df['macd_signal_line'], 'macd_signal'] = 1
    df.loc[df['macd'] < df['macd_signal_line'], 'macd_signal'] = -1
    return df

def calculate_adx_signals(df):
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Price'])
    df['ADX'] = adx.adx()
    df['DI+'] = adx.adx_pos()
    df['DI-'] = adx.adx_neg()
    df['adx_signal'] = 0
    df.loc[(df['DI+'] > df['DI-']) & (df['ADX'] > 10), 'adx_signal'] = 1
    df.loc[(df['DI+'] < df['DI-']) & (df['ADX'] > 10), 'adx_signal'] = -1
    return df

def plot_signals(df, indicator_name, signal_column):
    plt.figure(figsize=(20, 10))
    unique_months = df['Month'].unique()
    month_indices = [df[df['Month'] == month].index[0] for month in unique_months]
    plt.plot(df.index, df['Price'], label='Price', alpha=0.7, linewidth=2)
    plt.scatter(df[df[signal_column] == 1].index,
                df[df[signal_column] == 1]['Price'],
                marker='^', color='green', label='Up Signal', s=150)
    plt.scatter(df[df[signal_column] == -1].index,
                df[df[signal_column] == -1]['Price'],
                marker='v', color='red', label='Down Signal', s=150)
    plt.title(f'{indicator_name} Signals', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.xticks(month_indices, unique_months, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
    

############Helper function for 2nd tab#####################    

def prepare_data_prophet(df):
    prophet_df = df[['Month', 'Palm oil']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%b-%y')
    prophet_df['covid'] = ((prophet_df['ds'] >= '2020-03-01') &
                          (prophet_df['ds'] <= '2021-08-01')).astype(int)
    prophet_df['war'] = ((prophet_df['ds'] >= '2021-12-01') &
                        (prophet_df['ds'] <= '2022-08-01')).astype(int)
    prophet_df = prophet_df.reset_index(drop=True)
    return prophet_df

def prepare_data_rf(df):
    rf_df = df.copy()
    for col in ['Palm oil', 'Rapeseed oil', 'Sunflower oil', 'Coconut oil']:
        rf_df[f'{col}_Lag1'] = rf_df[col].shift(1)
    rf_df['Price_Ratio_Palm_Rapeseed'] = rf_df['Palm oil_Lag1'] / rf_df['Rapeseed oil_Lag1']
    rf_df['Price_Ratio_Palm_Sunflower'] = rf_df['Palm oil_Lag1'] / rf_df['Sunflower oil_Lag1']
    rf_df['Price_Ratio_Palm_Coconut'] = rf_df['Palm oil_Lag1'] / rf_df['Coconut oil_Lag1']
    rf_df['Month'] = pd.to_datetime(rf_df['Month'], format='%b-%y')
    rf_df = rf_df.dropna().reset_index(drop=True)
    return rf_df

# Train RF Model with Hyperparameter Tuning
def train_rf_model(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Train XGBoost Model with Hyperparameter Tuning
def train_xgb_model(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Train Prophet Model
def train_prophet_model(df_prophet):
    model = Prophet()
    model.add_regressor('covid')
    model.add_regressor('war')
    model.fit(df_prophet)
    return model

def calculate_rmse_percentage(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted)) / np.mean(actual) * 100

def calculate_directional_accuracy(actual, predicted):
    return np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100


# Main Streamlit App
st.title("Trading Strategy & Price Prediction")
tabs = st.tabs(["Trading Strategy Analyzer", "Palm Oil Price Prediction"])

# Tab 1: Trading Strategy Analyzer
with tabs[0]:
    st.header("Trading Strategy Analyzer")
    uploaded_file = st.file_uploader("Upload a CSV file for trading data", type=["csv"], key="trading_tab")
    if uploaded_file:
        df_data = load_and_prepare_data(uploaded_file)
        df_macd = calculate_macd_signals(df_data.copy())
        df_sto = calculate_stochastic_signals(df_data.copy())
        df_adx = calculate_adx_signals(df_data.copy())
        st.subheader("MACD Signals")
        plot_signals(df_macd, 'MACD', 'macd_signal')
        st.subheader("Stochastic Signals")
        plot_signals(df_sto, 'Stochastic', 'stoch_signal')
        st.subheader("ADX Signals")
        plot_signals(df_adx, 'ADX', 'adx_signal')
        success_days_min = 3
        success_days_max = 7
        success_threshold = 0.02

        # Evaluate success rates
        st.header("Strategy Success Rates")
        indicators = [
            (df_macd, 'MACD', 'macd_signal'),
            (df_sto, 'Stochastic', 'stoch_signal'),
            (df_adx, 'ADX', 'adx_signal')
        ]

        for df, indicator_name, signal_column in indicators:
            success_rate, success_count, total_signals, details = evaluate_strategy_success(
                df, signal_column, success_days_min, success_days_max, success_threshold
            )
            st.subheader(f"{indicator_name} Strategy")
            st.write(f"Total Signals: {total_signals}")
            st.write(f"Successful Signals: {success_count}")
            st.write(f"Success Rate: {success_rate:.2f}%")

# Tab 2: Combined Prediction
with tabs[1]:
    st.title("Palm Oil Price Prediction")
    st.write("An interactive dashboard for predicting palm oil prices using Random Forest, XGBoost, and Prophet.")

    # File Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

        # Prepare Data
        prophet_df = prepare_data_prophet(df)
        rf_df = prepare_data_rf(df)

        # Define Features
        numerical_features = [
            'Palm oil_Lag1', 'Rapeseed oil_Lag1', 'Sunflower oil_Lag1', 'Coconut oil_Lag1',
            'Price_Ratio_Palm_Rapeseed', 'Price_Ratio_Palm_Sunflower', 'Price_Ratio_Palm_Coconut'
        ]

        # Split Data
        train_data = rf_df[rf_df['Month'] < '2024-07-01']
        test_data = rf_df[(rf_df['Month'] >= '2024-07-01') & (rf_df['Month'] <= '2024-12-31')]
        X_train, y_train = train_data[numerical_features], train_data['Palm oil']
        X_test, y_test = test_data[numerical_features], test_data['Palm oil']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Models
        st.write("Training models...")
        rf_model = train_rf_model(X_train_scaled, y_train)
        xgb_model = train_xgb_model(X_train_scaled, y_train)
        prophet_model = train_prophet_model(prophet_df[prophet_df['ds'] < '2024-07-01'])

        # Predictions
        rf_test_predictions = rf_model.predict(X_test_scaled)
        xgb_test_predictions = xgb_model.predict(X_test_scaled)

        future_dates = pd.date_range('2024-07-01', '2024-12-31', freq='M')
        future_prophet = pd.DataFrame({'ds': future_dates})
        future_prophet['covid'] = 0
        future_prophet['war'] = 0
        prophet_forecast = prophet_model.predict(future_prophet)
        prophet_test_predictions = prophet_forecast['yhat'].values

        # Calculate Metrics
        rf_r2 = r2_score(y_test, rf_test_predictions)
        xgb_r2 = r2_score(y_test, xgb_test_predictions)
        prophet_r2 = r2_score(y_test, prophet_test_predictions)

        rf_rmse_percentage = calculate_rmse_percentage(y_test, rf_test_predictions)
        xgb_rmse_percentage = calculate_rmse_percentage(y_test, xgb_test_predictions)
        prophet_rmse_percentage = calculate_rmse_percentage(y_test, prophet_test_predictions)

        rf_directional_accuracy = calculate_directional_accuracy(y_test, rf_test_predictions)
        xgb_directional_accuracy = calculate_directional_accuracy(y_test, xgb_test_predictions)
        prophet_directional_accuracy = calculate_directional_accuracy(y_test, prophet_test_predictions)
        
        # Create a DataFrame for Model Accuracy Metrics
        metrics_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Prophet'],
            'R²': [rf_r2, xgb_r2, prophet_r2],
            'RMSE Percentage': [rf_rmse_percentage, xgb_rmse_percentage, prophet_rmse_percentage],
            'Directional Accuracy': [rf_directional_accuracy, xgb_directional_accuracy, prophet_directional_accuracy]
        })
        
        # Custom CSS to adjust column width
        st.markdown("""
            <style>
            .stDataFrame table th, .stDataFrame table td {
                padding: 10px;
                text-align: center;
            }
            .stDataFrame table th:nth-child(1), .stDataFrame table td:nth-child(1) {
                width: 200px;
            }
            .stDataFrame table th:nth-child(2), .stDataFrame table td:nth-child(2) {
                width: 100px;
            }
            .stDataFrame table th:nth-child(3), .stDataFrame table td:nth-child(3) {
                width: 150px;
            }
            .stDataFrame table th:nth-child(4), .stDataFrame table td:nth-child(4) {
                width: 200px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display Model Accuracy Metrics in Tabular Format
        st.write("### Model Accuracy Metrics:")
        st.dataframe(metrics_df)


        # Test Data Predictions
        st.write("### Test Data Predictions:")
        test_results = pd.DataFrame({
            'Month': test_data['Month'],
            'Actual Price': y_test.values,
            'Prophet Prediction': prophet_test_predictions,
            'Random Forest Prediction': rf_test_predictions,
            'XGBoost Prediction': xgb_test_predictions
        })
        st.dataframe(test_results)

        # Monthly and Quarterly Aggregation
        test_results['Month'] = pd.to_datetime(test_results['Month'])
        test_results['Quarter'] = test_results['Month'].dt.to_period('Q')

        monthly_results = test_results.groupby('Month').mean()
        quarterly_results = test_results.groupby('Quarter').mean()

        st.write("### Monthly Aggregation")
        st.dataframe(monthly_results)

        st.write("### Quarterly Aggregation")
        st.dataframe(quarterly_results)

        # Plot Predictions
        st.write("### Predictions Visualization")
        plt.figure(figsize=(10, 6))
        plt.plot(test_data['Month'], y_test, label='Actual Price', color='black', marker='o')
        plt.plot(test_data['Month'], rf_test_predictions, label='Random Forest', color='green', linestyle='--')
        plt.plot(test_data['Month'], xgb_test_predictions, label='XGBoost', color='red', linestyle='--')
        plt.plot(test_data['Month'], prophet_test_predictions, label='Prophet', color='blue', linestyle='--')
        plt.legend()
        plt.title("Palm Oil Price Predictions (July-Dec 2024)")
        plt.xlabel("Month")
        plt.ylabel("Price")
        plt.grid(True)
        st.pyplot(plt)

        # Download Results
        st.write("### Download Predictions")
        csv = test_results.to_csv(index=False)
        st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
