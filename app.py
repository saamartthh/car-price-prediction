import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for unique styling
st.markdown("""
    <style>
    .main-title {
        font-size: 45px !important;
        font-weight: bold;
        color: #FF6347; /* Tomato color */
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        font-size: 30px !important;
        font-weight: bold;
        color: #4CAF50; /* Green */
        background: linear-gradient(to right, #e0f7fa, #f1f8e9);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #f3e5f5, #e1bee7);
        color: black;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #C70039;
    }
    .subheading {
        font-size: 25px !important;
        font-weight: bold;
        color: #6A1B9A; /* Purple */
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and data
model = pk.load(open('model.pkl', 'rb'))
cars_data = pd.read_csv('Cardetails.csv')

# Process car brand names
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Sidebar with instructions
st.sidebar.title("Welcome to Car Price Predictor 🚘")
st.sidebar.info("📝 Fill in the car details below to predict the price accurately.")
# st.sidebar.image("https://cdn.pixabay.com/photo/2018/01/09/21/10/car-3075936_960_720.jpg", use_container_width=True)

# Main title
st.markdown('<p class="main-title">Car Price Prediction Tool</p>', unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

# Left Column: Car Details
with col1:
    st.markdown('<p class="subheading">Car Details</p>', unsafe_allow_html=True)
    name = st.selectbox('🚗 Select Car Brand', cars_data['name'].unique())

    # Year input as number_input
    year = st.number_input('📅 Year of Manufacture', min_value=1994, max_value=2024, value=2010)

    # Kilometers driven as number_input
    km_driven = st.number_input('🛣️ Kilometers Driven', min_value=11, max_value=200000, value=50000, step=1000)

    # Fuel type
    fuel = st.radio('⛽ Fuel Type', cars_data['fuel'].unique())

    # Seller type
    seller_type = st.selectbox('👤 Seller Type', cars_data['seller_type'].unique())

# Right Column: Technical Specifications
with col2:
    st.markdown('<p class="subheading">Technical Specifications</p>', unsafe_allow_html=True)

    # Transmission
    transmission = st.radio('🔧 Transmission Type', cars_data['transmission'].unique())

    # Owner type
    owner = st.selectbox('🔑 Owner Type', cars_data['owner'].unique())

    # Mileage as number_input
    mileage = st.number_input('⚡ Mileage (kmpl)', min_value=10, max_value=40, value=20, step=1)

    # Engine as number_input
    engine = st.number_input('🔋 Engine (CC)', min_value=700, max_value=5000, value=1500, step=100)

    # Max Power as number_input
    max_power = st.number_input('💪 Max Power (bhp)', min_value=0, max_value=200, value=100, step=1)

    # Number of seats
    seats = st.selectbox('💺 Number of Seats', [5, 6, 7, 8, 9, 10])

# Center Predict Button
col_center = st.columns([2, 1, 2])[1]
with col_center:
    predict_button = st.button("🎯 Predict Car Price")

# Prediction Logic
if predict_button:
    # Show spinner during prediction
    with st.spinner('Calculating...'):
        # Prepare input data for the model
        input_data = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )

        # Map categorical data to numerical values
        input_data['owner'].replace(
            ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], 
            [1, 2, 3, 4, 5], inplace=True
        )
        input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
        input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
        input_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
        input_data['name'].replace(
            ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 
             'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 
             'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 
             'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
            list(range(1, 32)), inplace=True
        )

        # Make prediction
        price = model.predict(input_data)[0]

        # Display prediction
        formatted_price = f"₹{price:,.2f}"
        st.markdown(f'<p class="prediction-box">Predicted Car Price:<br>{formatted_price}</p>', unsafe_allow_html=True)

        # Show input summary
        with st.expander("See Your Inputs"):
            st.json({
                "Car Brand": name,
                "Year": year,
                "Kilometers Driven": f"{km_driven:,} km",
                "Fuel Type": fuel,
                "Seller Type": seller_type,
                "Transmission": transmission,
                "Owner Type": owner,
                "Mileage": f"{mileage} kmpl",
                "Engine": f"{engine} CC",
                "Max Power": f"{max_power} bhp",
                "Seats": seats
            })

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 *This tool uses historical data to predict car prices accurately!*")
