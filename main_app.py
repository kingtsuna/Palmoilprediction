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
                marker='v', color='green', label='Buy Signal', s=150)
    plt.scatter(df[df[signal_column] == -1].index,
                df[df[signal_column] == -1]['Price'],
                marker='^', color='red', label='Sell Signal', s=150)
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

# # Main Streamlit App
# st.title("Trading Strategy & Price Prediction")
# tabs = st.tabs(["Trading Strategy Analyzer", "Palm Oil Price Prediction"])

# with tabs[0]:
#     st.header("Trading Strategy Analyzer")
#     uploaded_file = st.file_uploader("Upload a CSV file for trading data", type=["csv"])
#     if uploaded_file:
#         df_data = load_and_prepare_data(uploaded_file)
#         df_macd = calculate_macd_signals(df_data.copy())
#         df_sto = calculate_stochastic_signals(df_data.copy())
#         df_adx = calculate_adx_signals(df_data.copy())
#         st.subheader("MACD Signals")
#         plot_signals(df_macd, 'MACD', 'macd_signal')
#         st.subheader("Stochastic Signals")
#         plot_signals(df_sto, 'Stochastic', 'stoch_signal')
#         st.subheader("ADX Signals")
#         plot_signals(df_adx, 'ADX', 'adx_signal')
#         success_days_min = 3
#         success_days_max = 7
#         success_threshold = 0.02

#         # Evaluate success rates
#         st.header("Strategy Success Rates")
#         indicators = [
#             (df_macd, 'MACD', 'macd_signal'),
#             (df_sto, 'Stochastic', 'stoch_signal'),
#             (df_adx, 'ADX', 'adx_signal')
#         ]

#         for df, indicator_name, signal_column in indicators:
#             success_rate, success_count, total_signals, details = evaluate_strategy_success(
#                 df, signal_column, success_days_min, success_days_max, success_threshold
#             )
#             st.subheader(f"{indicator_name} Strategy")
#             st.write(f"Total Signals: {total_signals}")
#             st.write(f"Successful Signals: {success_count}")
#             st.write(f"Success Rate: {success_rate:.2f}%")

# with tabs[1]:
#     st.title("Palm Oil Price Prediction")

#     # st.sidebar.header("Upload your CSV file")
#     uploaded_file2 = st.file_uploader("Upload a CSV file for trading data", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file2)

#         # Prepare the data
#         df_prepared = prepare_data(df)

#         # Create features and target
#         X, y, dates = create_features_target(df_prepared)

#         # Split the data
#         X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
#             X, y, dates, test_size=0.2, random_state=42
#         )

#         # Train the model
#         model = RandomForestRegressor(
#             n_estimators=300,
#             max_depth=10,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             max_features='sqrt',
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred_train = model.predict(X_train)
#         y_pred_test = model.predict(X_test)

#         # Calculate metrics
#         mse = mean_squared_error(y_test, y_pred_test)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, y_pred_test)

#         st.write("### Model Performance Metrics")
#         st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#         st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#         st.write(f"**R-squared Score (R²):** {r2:.2f}")

#         # Calculate feature importance
#         feature_importance = pd.DataFrame({
#             'feature': X.columns,
#             'importance': model.feature_importances_
#         }).sort_values('importance', ascending=False)

#         st.write("### Feature Importance")
#         st.write(feature_importance)

#         # Plot feature importance
#         plot_feature_importance(feature_importance)

#         # Plot training and test predictions
#         train_data = pd.DataFrame({
#             'date': dates_train,
#             'actual': y_train,
#             'predicted': y_pred_train
#         }).sort_values('date')

#         test_data = pd.DataFrame({
#             'date': dates_test,
#             'actual': y_test,
#             'predicted': y_pred_test
#         }).sort_values('date')

#         st.write("### Training Set Predictions")
#         plot_time_series(train_data['date'], train_data['actual'], train_data['predicted'], 
#                             "Palm Oil Price Over Time - Training Set")

#         st.write("### Test Set Predictions")
#         plot_time_series(test_data['date'], test_data['actual'], test_data['predicted'], 
#                             "Palm Oil Price Over Time - Test Set")

#         # Predict next month's price
#         last_month_data = X.iloc[[-1]]
#         next_month_prediction = model.predict(last_month_data)
#         st.write(f"### Predicted Palm Oil Price for Next Month: {next_month_prediction[0]:.2f}")
#     else:
#         st.info("Please upload a CSV file to proceed.")

# # Main Streamlit App
# st.title("Trading Strategy & Price Prediction")
# tabs = st.tabs(["Trading Strategy Analyzer", "Palm Oil Price Prediction"])

# # Tab 1: Trading Strategy Analyzer
# with tabs[0]:
#     st.header("Trading Strategy Analyzer")
#     uploaded_file = st.file_uploader("Upload a CSV file for trading data", type=["csv"], key="trading_tab")
#     if uploaded_file:
#         df_data = load_and_prepare_data(uploaded_file)
#         df_macd = calculate_macd_signals(df_data.copy())
#         df_sto = calculate_stochastic_signals(df_data.copy())
#         df_adx = calculate_adx_signals(df_data.copy())
#         st.subheader("MACD Signals")
#         plot_signals(df_macd, 'MACD', 'macd_signal')
#         st.subheader("Stochastic Signals")
#         plot_signals(df_sto, 'Stochastic', 'stoch_signal')
#         st.subheader("ADX Signals")
#         plot_signals(df_adx, 'ADX', 'adx_signal')
#         success_days_min = 3
#         success_days_max = 7
#         success_threshold = 0.02

#         # Evaluate success rates
#         st.header("Strategy Success Rates")
#         indicators = [
#             (df_macd, 'MACD', 'macd_signal'),
#             (df_sto, 'Stochastic', 'stoch_signal'),
#             (df_adx, 'ADX', 'adx_signal')
#         ]

#         for df, indicator_name, signal_column in indicators:
#             success_rate, success_count, total_signals, details = evaluate_strategy_success(
#                 df, signal_column, success_days_min, success_days_max, success_threshold
#             )
#             st.subheader(f"{indicator_name} Strategy")
#             st.write(f"Total Signals: {total_signals}")
#             st.write(f"Successful Signals: {success_count}")
#             st.write(f"Success Rate: {success_rate:.2f}%")

# # Tab 2: Palm Oil Price Prediction
# with tabs[1]:
#     st.title("Palm Oil Price Prediction")

#     # Unique key for the second file uploader
#     uploaded_file2 = st.file_uploader("Upload a CSV file for palm oil price prediction", type=["csv"], key="prediction_tab")

#     if uploaded_file2 is not None:
#         df = pd.read_csv(uploaded_file2)

#         # Prepare the data
#         df_prepared = prepare_data(df)

#         # Create features and target
#         X, y, dates = create_features_target(df_prepared)

#         # Split the data
#         X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
#             X, y, dates, test_size=0.2, random_state=42
#         )

#         # Train the model
#         model = RandomForestRegressor(
#             n_estimators=300,
#             max_depth=10,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             max_features='sqrt',
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred_train = model.predict(X_train)
#         y_pred_test = model.predict(X_test)

#         # Calculate metrics
#         mse = mean_squared_error(y_test, y_pred_test)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, y_pred_test)

#         st.write("### Model Performance Metrics")
#         st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#         st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#         st.write(f"**R-squared Score (R²):** {r2:.2f}")

#         # Calculate feature importance
#         feature_importance = pd.DataFrame({
#             'feature': X.columns,
#             'importance': model.feature_importances_
#         }).sort_values('importance', ascending=False)

#         st.write("### Feature Importance")
#         st.write(feature_importance)

#         # Plot feature importance
#         plot_feature_importance(feature_importance)

#         # Plot training and test predictions
#         train_data = pd.DataFrame({
#             'date': dates_train,
#             'actual': y_train,
#             'predicted': y_pred_train
#         }).sort_values('date')

#         test_data = pd.DataFrame({
#             'date': dates_test,
#             'actual': y_test,
#             'predicted': y_pred_test
#         }).sort_values('date')

#         st.write("### Training Set Predictions")
#         plot_time_series(train_data['date'], train_data['actual'], train_data['predicted'], 
#                             "Palm Oil Price Over Time - Training Set")

#         st.write("### Test Set Predictions")
#         plot_time_series(test_data['date'], test_data['actual'], test_data['predicted'], 
#                             "Palm Oil Price Over Time - Test Set")

#         # Predict next month's price
#         last_month_data = X.iloc[[-1]]
#         next_month_prediction = model.predict(last_month_data)
#         st.write(f"### Predicted Palm Oil Price for Next Month: {next_month_prediction[0]:.2f}")
#     else:
#         st.info("Please upload a CSV file to proceed.")

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


# Main Streamlit App
st.title("Trading Strategy & Price Prediction")
tabs = st.tabs(["Trading Strategy Analyzer", "Palm Oil Price Prediction", "Palm Oil Price Forecast with Prophet"])

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

# Tab 2: Palm Oil Price Prediction
with tabs[1]:
    st.title("Palm Oil Price Prediction")

    # Unique key for the second file uploader
    uploaded_file2 = st.file_uploader("Upload a CSV file for palm oil price prediction", type=["csv"], key="prediction_tab")

    if uploaded_file2 is not None:
        df = pd.read_csv(uploaded_file2)

        # Prepare the data
        df_prepared = prepare_data(df)

        # Create features and target
        X, y, dates = create_features_target(df_prepared)

        # Split the data
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.2, random_state=42
        )

        # Train the model
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        st.write("### Model Performance Metrics")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared Score (R²):** {r2:.2f}")

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        st.write("### Feature Importance")
        st.write(feature_importance)

        # Plot feature importance
        plot_feature_importance(feature_importance)

        # Plot training and test predictions
        train_data = pd.DataFrame({
            'date': dates_train,
            'actual': y_train,
            'predicted': y_pred_train
        }).sort_values('date')

        test_data = pd.DataFrame({
            'date': dates_test,
            'actual': y_test,
            'predicted': y_pred_test
        }).sort_values('date')

        st.write("### Training Set Predictions")
        plot_time_series(train_data['date'], train_data['actual'], train_data['predicted'], 
                            "Palm Oil Price Over Time - Training Set")

        st.write("### Test Set Predictions")
        plot_time_series(test_data['date'], test_data['actual'], test_data['predicted'], 
                            "Palm Oil Price Over Time - Test Set")

        # Predict next month's price
        last_month_data = X.iloc[[-1]]
        next_month_prediction = model.predict(last_month_data)
        st.write(f"### Predicted Palm Oil Price for Next Month: {next_month_prediction[0]:.2f}")
    else:
        st.info("Please upload a CSV file to proceed.")

# Tab 3: Palm Oil Price Forecast with Prophet
with tabs[2]:
    st.title("Palm Oil Price Forecast with Prophet")

    # File uploader for Prophet model
    uploaded_file_prophet = st.file_uploader("Upload a CSV file with 'Date' and 'Price' columns for Prophet", type=["csv"], key="prophet_tab")
    if uploaded_file_prophet:
        df = pd.read_csv(uploaded_file_prophet)

        # Data preparation for Prophet
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'], format='%b-%y')
        df['covid'] = ((df['ds'] >= '2020-03-01') & (df['ds'] <= '2021-08-01')).astype(int)
        df['war'] = ((df['ds'] >= '2021-12-01') & (df['ds'] <= '2022-08-01')).astype(int)

        # Train the Prophet model
        with st.spinner("Training the Prophet model..."):
            model = create_prophet_model(df)

        # Make predictions with Prophet
        with st.spinner("Making predictions..."):
            forecast, rmse, r2 = make_predictions(model, df)

        # Display metrics
        st.write("### Model Performance Metrics")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R-squared Score (R²):** {r2:.2f}")

        # Display predictions for next 6 months
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
        st.write("### Predicted Palm Oil Prices for the Next 6 Months")
        st.dataframe(predictions.rename(columns={
            'ds': 'Date',
            'yhat': 'Predicted Price',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        }))

        # Plot results
        st.write("### Forecast Plot")
        fig_forecast, fig_components = plot_results(model, forecast)
        st.pyplot(fig_forecast)
        st.write("### Forecast Components")
        st.pyplot(fig_components)

        # Save predictions to CSV
        predictions_df = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        predictions_df.columns = ['Date', 'Predicted_Price', 'Lower_Bound', 'Upper_Bound']
        predictions_csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=predictions_csv,
            file_name='palm_oil_predictions.csv',
            mime='text/csv'
        )
    else:
        st.info("Please upload a CSV file to proceed.")
