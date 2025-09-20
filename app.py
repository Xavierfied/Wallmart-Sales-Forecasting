import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import re
import unicodedata


# --- 1. Load Data and Model (IMPORTANT: These files must be in the same directory as this script) ---
# Use st.cache_data to load data only once for efficiency
@st.cache_data
def load_data_and_model():
    """Loads the training data and the pre-trained pipeline."""
    try:
        # Assumed file paths. Replace with your actual paths if different.
        training_data = pd.read_csv('full_training_dataset.csv')
        pipeline = joblib.load('xgb_forecasting_pipeline.joblib')
        return training_data, pipeline
    except FileNotFoundError as e:
        st.error(
            f"Error: Required file not found. Please ensure all files are in the same folder as the app script. Missing file: {e.filename}")
        return None, None


def create_features(df):
    """
    Creates the same time series features used for training the model.
    """
    df_copy = df.copy()
    if 'Weekly_Sales' not in df_copy.columns:
        df_copy['Weekly_Sales'] = 0

    for lag in [1, 2, 4, 52]:
        df_copy[f"lag_{lag}"] = df_copy.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)

    windows = [4, 12, 52]
    for w in windows:
        df_copy[f"rolling_mean_{w}"] = df_copy.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(w).mean()
        df_copy[f"rolling_std_{w}"] = df_copy.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(w).std()
        df_copy[f"rolling_min_{w}"] = df_copy.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(w).min()
        df_copy[f"rolling_max_{w}"] = df_copy.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(w).max()

    return df_copy


def find_column(df_columns, possible_names, column_label):
    """
    Finds the correct column name by performing aggressive cleaning on both the
    dataframe columns and the list of possible names.
    This version includes print statements for debugging.
    """
    st.info(f"Searching for the '{column_label}' column...")
    clean_possible_names = {re.sub(r'[^a-z0-9]', '', name.lower()) for name in possible_names}
    st.info(f"Cleaned names to search for: {clean_possible_names}")

    for col in df_columns:
        # Normalize unicode, encode to ascii, decode back, then remove non-alphanumeric chars
        clean_col = re.sub(r'[^a-z0-9]', '',
                           unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8').lower())
        st.info(f"Checking uploaded column '{col}' (cleaned: '{clean_col}')")
        if clean_col in clean_possible_names:
            st.success(f"Match found! Using column '{col}' for {column_label}.")
            return col

    st.error(f"No matching column found for '{column_label}' after extensive cleaning.")
    return None


# --- 2. Streamlit UI and Logic ---
st.set_page_config(page_title="Sales Forecast Tool", layout="wide")
st.title('Retail Sales Forecast Tool')
st.markdown(
    'Upload a CSV file with future dates to predict weekly sales. The app will generate a forecast chart and provide the data for download.')

# Load data and model only once
training_data, pipeline = load_data_and_model()

if training_data is not None and pipeline is not None:
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_to_predict = pd.read_csv(uploaded_file)

            # --- Robust Column Finding and Renaming ---
            st.info(f"Original columns found in uploaded file: {df_to_predict.columns.tolist()}")

            store_col = find_column(df_to_predict.columns, ['Store', 'Store ID', 'StoreId', 'store_id'], 'Store')
            dept_col = find_column(df_to_predict.columns, ['Dept', 'Department', 'Department ID', 'dept_id'], 'Dept')
            date_col = find_column(df_to_predict.columns, ['Date'], 'Date')

            # Validate that key columns exist
            if not all([store_col, dept_col, date_col]):
                missing = []
                if not store_col:
                    missing.append("'Store'")
                if not dept_col:
                    missing.append("'Dept'")
                if not date_col:
                    missing.append("'Date'")

                error_message = f"Error: Required columns are missing. The app could not find a column for: {', '.join(missing)}.\n\n"
                error_message += "Please ensure your CSV file has columns with one of the following names (case and space insensitive):\n"
                if not store_col:
                    error_message += "- For Store: 'Store', 'Store ID', 'StoreId', 'store_id'\n"
                if not dept_col:
                    error_message += "- For Dept: 'Dept', 'Department', 'Department ID', 'dept_id'\n"
                if not date_col:
                    error_message += "- For Date: 'Date'\n"

                st.error(error_message)

            else:
                # Rename columns to a consistent format for the app's logic
                df_to_predict.rename(columns={store_col: 'Store', dept_col: 'Dept', date_col: 'Date'}, inplace=True)

                # --- User Input ---
                col1, col2 = st.columns(2)
                with col1:
                    unique_stores = sorted(df_to_predict['Store'].unique())
                    selected_store = st.selectbox('Select Store ID', unique_stores, index=0)
                with col2:
                    unique_depts = sorted(df_to_predict['Dept'].unique())
                    selected_dept = st.selectbox('Select Department ID', unique_depts, index=0)

                if st.button('Generate Forecast'):
                    with st.spinner('Generating forecast...'):
                        # Filter data for the selected store and department
                        history_subset = training_data[
                            (training_data['Store'] == selected_store) &
                            (training_data['Dept'] == selected_dept)
                            ]
                        test_data = df_to_predict[
                            (df_to_predict['Store'] == selected_store) &
                            (df_to_predict['Dept'] == selected_dept)
                            ]

                        if history_subset.empty or test_data.empty:
                            st.warning("No data found for the selected Store/Department combination.")
                        else:
                            # Combine a small chunk of historical data with the new data to create features
                            combined_chunk = pd.concat([history_subset.tail(52), test_data])
                            combined_chunk = create_features(combined_chunk)

                            # Drop columns that are not features from the model
                            feature_columns = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI',
                                               'Unemployment', 'Type', 'Size',
                                               'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                                               'year', 'month', 'week_of_year',
                                               'lag_1', 'lag_2', 'lag_4', 'lag_52',
                                               'rolling_mean_4', 'rolling_std_4', 'rolling_min_4', 'rolling_max_4',
                                               'rolling_mean_12', 'rolling_std_12', 'rolling_min_12', 'rolling_max_12',
                                               'rolling_mean_52', 'rolling_std_52', 'rolling_min_52', 'rolling_max_52']

                            # Ensure the order of columns matches the training data
                            X_future = combined_chunk.loc[test_data.index][feature_columns]

                            # Make the prediction
                            y_future_pred = pipeline.predict(X_future)

                            # Add the predictions back to the test data
                            test_data['Predicted_Weekly_Sales'] = y_future_pred

                            # --- Display Results ---
                            st.success("Forecast generated successfully!")

                            # Create a layout with two columns
                            col3, col4 = st.columns([2, 1])

                            with col3:
                                # Plot the sales forecast
                                history_subset['Date'] = pd.to_datetime(history_subset['Date'])
                                history_subset = history_subset.set_index('Date')
                                test_data['Date'] = pd.to_datetime(test_data['Date'])
                                test_data = test_data.set_index('Date')

                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(history_subset.index, history_subset['Weekly_Sales'], label='Historical Sales',
                                        color='dodgerblue')
                                ax.plot(test_data.index, test_data['Predicted_Weekly_Sales'], label='Predicted Sales',
                                        color='red', linestyle='--')
                                ax.axvline(x=history_subset.index[-1], color='grey', linestyle=':',
                                           label='Prediction Start')

                                ax.set_title(f"Sales Forecast for Store {selected_store}, Dept {selected_dept}",
                                             fontsize=16)
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Weekly Sales ($)")
                                ax.legend()
                                ax.grid(True)
                                plt.xticks(rotation=45)
                                st.pyplot(fig)

                            with col4:
                                st.subheader("Predicted Data")
                                st.dataframe(test_data[['Predicted_Weekly_Sales', 'Date']].reset_index(drop=True),
                                             use_container_width=True)

                                csv_buffer = io.StringIO()
                                test_data[['Date', 'Predicted_Weekly_Sales']].to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()

                                st.download_button(
                                    label="Download Predicted Sales CSV",
                                    data=csv_data,
                                    file_name=f"forecast_Store{selected_store}_Dept{selected_dept}.csv",
                                    mime="text/csv"
                                )

        except KeyError as e:
            st.error(f"An error occurred. Please check the columns in your file. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
