import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model and Data ---
model = joblib.load('flight_demand_predictor.joblib')
df = pd.read_csv('data/customer_booking.csv', encoding='latin1')
df_completed = df[df['booking_complete'] == 1]

# --- 2. Replicate the "Rare Route" Logic ---
master_df_for_counts = df_completed.groupby(['route', 'flight_day']).agg(
    total_passengers=('num_passengers', 'sum')
).reset_index()
route_counts = master_df_for_counts['route'].value_counts()
threshold = 5
rare_routes = route_counts[route_counts < threshold].index.tolist()

# --- 3. User Interface Setup ---
ROUTE_DECODER = {
    'AKLKUL': 'Auckland (AKL) to Kuala Lumpur (KUL)',
    'PENTPE': 'Penang (PEN) to Taipei (TPE)',
    'DMKIXB': 'Don Mueang (DMK) to Bagdogra (IXB)',
    'ICNCTS': 'Seoul (ICN) to Sapporo (CTS)',
    'CTSSIN': 'Sapporo (CTS) to Singapore (SIN)'
}
def format_route(route_code):
    return ROUTE_DECODER.get(route_code, route_code)

st.set_page_config(page_title="Flight Demand Predictor", layout="wide")
st.title('✈️ Airline Passenger Demand Forecasting')
st.markdown("Select a flight route and day of the week to forecast the total number of passengers.")

st.sidebar.header('Make a Prediction')
route_list = sorted(df_completed['route'].unique())
day_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

selected_route = st.sidebar.selectbox('Route', route_list, format_func=format_route)
selected_day = st.sidebar.selectbox('Flight Day', day_list)

# --- 4. Prediction Logic ---
if st.sidebar.button('Predict Demand'):
    if selected_route in rare_routes:
        route_for_model = 'Other'
    else:
        route_for_model = selected_route
    
    route_data = df_completed[df_completed['route'] == selected_route]
    if not route_data.empty:
        avg_duration = route_data['flight_duration'].mean()
        haul_type = 'Long' if avg_duration > 6 else 'Short'

        input_data = {
            'route': [route_for_model],
            'flight_day': [selected_day],
            'avg_flight_duration': [avg_duration],
            'haul_type': [haul_type]
        }
        input_df = pd.DataFrame(input_data)

        prediction = model.predict(input_df)
        predicted_passengers = round(prediction[0])

        st.subheader(f'Predicted Demand for: {format_route(selected_route)}')
        st.metric(label="Estimated Passengers", value=f"{predicted_passengers}")

        if predicted_passengers > 70:
            st.success("High demand expected.")
        elif predicted_passengers < 50:
            st.warning("Low demand expected.")
        else:
            st.info("Moderate demand expected.")
    else:
        st.error(f"No data available for the selected route: {selected_route}")
else:
    st.info('Select a route and day from the sidebar and click "Predict Demand" to see the forecast.')

st.sidebar.markdown("---")
st.sidebar.info("This app uses a Random Forest Regressor trained on historical booking data.")

# --- NEW: Add Model Performance Metrics ---
st.sidebar.markdown("### Model Performance")
st.sidebar.markdown(f"**$R^2$ Score:** 0.79")
st.sidebar.markdown(f"**Mean Absolute Error:** 2.36 passengers")

# --- NEW: Add Your Name at the Bottom ---
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        Made by Syed Abdul Waheed
    </div>
    """,
    unsafe_allow_html=True
)