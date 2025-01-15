import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import warnings
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

# Palm Oil Price Prediction Functions
def prepare_data(df):
    df['Palm oil_Lag1'] = df['Palm oil'].shift(1)
    df['Rapeseed oil_Lag1'] = df['Rapeseed oil'].shift(1)
    df['Sunflower oil_Lag1'] = df['Sunflower oil'].shift(1)
    df['Coconut oil_Lag1'] = df['Coconut oil'].shift(1)
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
    df['Price_Ratio_Palm_Rapeseed'] = df['Palm oil_Lag1'] / df['Rapeseed oil_Lag1']
    df['Price_Ratio_Palm_Sunflower'] = df['Palm oil_Lag1'] / df['Sunflower oil_Lag1']
    df['Price_Ratio_Palm_Coconut'] = df['Palm oil_Lag1'] / df['Coconut oil_Lag1']
    return df.dropna()

def create_features_target(df):
    feature_columns = [
        'Palm oil_Lag1', 'Rapeseed oil_Lag1', 'Sunflower oil_Lag1',
        'Coconut oil_Lag1', 'Price_Ratio_Palm_Rapeseed',
        'Price_Ratio_Palm_Sunflower', 'Price_Ratio_Palm_Coconut'
    ]
    X = df[feature_columns]
    y = df['Palm oil']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y, df['Month']

def plot_time_series(dates, y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates, y_true, label='Actual', marker='o', alpha=0.7)
    ax.plot(dates, y_pred, label='Predicted', marker='x', alpha=0.7)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_feature_importance(feature_importance):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    ax.set_title("Feature Importance", fontsize=14, pad=20)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    st.pyplot(fig)


def create_prophet_model(df):
    """
    Create and fit Prophet model with optimized parameters
    """
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=0.01,
        holidays_prior_scale=0.01,
        seasonality_mode='additive',
        changepoint_range=0.95
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    model.add_seasonality(name='quarterly', period=365.25 / 4, fourier_order=5)
    model.add_regressor('covid')
    model.add_regressor('war')
    model.fit(df)
    return model


def make_predictions(model, df):
    """
    Make predictions and calculate metrics
    """
    future = model.make_future_dataframe(periods=6, freq='M')
    future['covid'] = ((future['ds'] >= '2020-03-01') & (future['ds'] <= '2021-08-01')).astype(int)
    future['war'] = ((future['ds'] >= '2021-12-01') & (future['ds'] <= '2022-08-01')).astype(int)
    forecast = model.predict(future)
    y_true = df['y'].values
    y_pred = forecast['yhat'].values[:len(df)]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return forecast, rmse, r2

def plot_results(model, forecast, title="Palm Oil Price Forecast"):
    """
    Create visualization of the forecast
    """
    fig_forecast = model.plot(forecast)
    fig_components = model.plot_components(forecast)
    return fig_forecast, fig_components
# Helper Functions
def calculate_rmse_percentage(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_actual = np.mean(y_true)
    rmse_percentage = (rmse / mean_actual) * 100
    return rmse_percentage

def calculate_directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_predictions = len(y_true) - 1
    directional_correct = sum((y_true[t + 1] - y_true[t]) * (y_pred[t + 1] - y_true[t]) > 0 for t in range(total_predictions))
    accuracy = (directional_correct / total_predictions) * 100
    return accuracy

def prepare_data_prophet(df):
    prophet_df = df[['Month', 'Palm oil']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%b-%y')
    prophet_df['covid'] = ((prophet_df['ds'] >= '2020-03-01') & 
                          (prophet_df['ds'] <= '2021-08-01')).astype(int)
    prophet_df['war'] = ((prophet_df['ds'] >= '2021-12-01') & 
                        (prophet_df['ds'] <= '2022-08-01')).astype(int)
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

def train_prophet_model(df):
    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1,
                    holidays_prior_scale=0.1, seasonality_mode='additive', changepoint_range=0.9)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)
    model.add_regressor('covid')
    model.add_regressor('war')
    model.fit(df)
    return model

def train_rf_model(X, y):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2,
                                   min_samples_leaf=1, max_features='sqrt', random_state=42)
    model.fit(X, y)
    return model

def combine_predictions(prophet_pred, rf_pred):
    prophet_weight = 0.05
    rf_weight = 0.95
    combined_pred = prophet_weight * prophet_pred + rf_weight * rf_pred
    return combined_pred

def plot_predictions(dates, y_true, y_pred_prophet, y_pred_rf, y_pred_combined):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates, y_true, label='Actual', marker='o', alpha=0.7)
    ax.plot(dates, y_pred_prophet, label='Prophet (Univariate)', marker='x', alpha=0.5)
    ax.plot(dates, y_pred_rf, label='Random Forest (Multivariate)', marker='+', alpha=0.5)
    ax.plot(dates, y_pred_combined, label='Combined Model', marker='*', alpha=0.7, linewidth=2)
    ax.set_title("Palm Oil Price Predictions", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


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
    # Streamlit App
    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file with palm oil data:", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        prophet_df = prepare_data_prophet(df)
        rf_df = prepare_data_rf(df)

        # st.write("### Data Preview")
        # st.dataframe(df.head())

        # Train Prophet Model
        # st.write("### Training Prophet Model")
        prophet_model = train_prophet_model(prophet_df)
        future_prophet = prophet_model.make_future_dataframe(periods=6, freq='M')
        future_prophet['covid'] = ((future_prophet['ds'] >= '2020-03-01') & 
                                (future_prophet['ds'] <= '2021-08-01')).astype(int)
        future_prophet['war'] = ((future_prophet['ds'] >= '2021-12-01') & 
                                (future_prophet['ds'] <= '2022-08-01')).astype(int)
        forecast_prophet = prophet_model.predict(future_prophet)

        # Train Random Forest Model
        # st.write("### Training Random Forest Model")
        numerical_features = [
            'Palm oil_Lag1', 'Rapeseed oil_Lag1', 'Sunflower oil_Lag1', 'Coconut oil_Lag1',
            'Price_Ratio_Palm_Rapeseed', 'Price_Ratio_Palm_Sunflower', 'Price_Ratio_Palm_Coconut'
        ]
        X = rf_df[numerical_features]
        y = rf_df['Palm oil']
        dates = rf_df['Month']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        rf_model = train_rf_model(X_scaled, y)

        # Generate Predictions
        prophet_predictions = forecast_prophet['yhat'].values[:len(prophet_df)]
        rf_predictions = rf_model.predict(X_scaled)
        min_length = min(len(y), len(prophet_predictions), len(rf_predictions))
        y_aligned = y[:min_length]
        prophet_pred_aligned = prophet_predictions[:min_length]
        rf_pred_aligned = rf_predictions[:min_length]
        dates_aligned = dates[:min_length]
        combined_predictions = combine_predictions(prophet_pred_aligned, rf_pred_aligned)

        # Metrics
        st.write("### Model Performance Metrics")
        metrics_data = {
            'Metric': ['Explained Fit', 'Prediction Error', 'Directional Accuracy'],
            'Prophet': [
                f"{r2_score(y_aligned, prophet_pred_aligned):.3f}",
                f"{calculate_rmse_percentage(y_aligned, prophet_pred_aligned):.2f}%",
                f"{calculate_directional_accuracy(y_aligned, prophet_pred_aligned):.2f}%"
            ],
            'Random Forest': [
                f"{r2_score(y_aligned, rf_pred_aligned):.3f}",
                f"{calculate_rmse_percentage(y_aligned, rf_pred_aligned):.2f}%",
                f"{calculate_directional_accuracy(y_aligned, rf_pred_aligned):.2f}%"
            ],
            'Combined': [
                f"{r2_score(y_aligned, combined_predictions):.3f}",
                f"{calculate_rmse_percentage(y_aligned, combined_predictions):.2f}%",
                f"{calculate_directional_accuracy(y_aligned, combined_predictions):.2f}%"
            ]
        }
        st.table(pd.DataFrame(metrics_data))

        # Plot Predictions
        st.write("### Predictions vs Actual Values")
        plot_predictions(dates_aligned, y_aligned, prophet_pred_aligned, rf_pred_aligned, combined_predictions)
